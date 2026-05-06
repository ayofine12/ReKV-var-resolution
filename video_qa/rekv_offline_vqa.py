import torch
from logzero import logger

from video_qa.base import BaseVQA, work


class ReKVOfflineVQA(BaseVQA):
    def video_open_qa(self, question, max_new_tokens=1024, retrieved_indices=None):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }

        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens, retrieved_indices=retrieved_indices)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    def video_close_qa(self, question, candidates, correct_choice, retrieved_indices=None):
        input_text = self.format_mcqa_prompt(question, candidates)
        choice_scores = {}
        if hasattr(self.qa_model, "multiple_choice_answering"):
            mc_kwargs = {
                'num_choices': len(candidates),
                'retrieved_indices': retrieved_indices,
            }
            if self.save_choice_scores:
                mc_kwargs['return_scores'] = True
            mc_output = self.qa_model.multiple_choice_answering(
                input_text,
                **mc_kwargs,
            )
            if isinstance(mc_output, dict):
                pred_answer = mc_output.get('pred_answer', mc_output.get('pred_choice', ''))
                choice_scores = {
                    key: value
                    for key, value in mc_output.items()
                    if key not in {'pred_answer', 'pred_choice'}
                }
            else:
                pred_answer = mc_output
        else:
            pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_answer = str(pred_answer).replace('\n', '')
        pred_letter = self.extract_characters_regex(pred_answer)
        result = {
            'pred_answer': pred_answer,
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }
        result.update(choice_scores)
        return result

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # load and preprocess video frames for QA
        video_path = video_sample['video_path']
        video = self.load_video(video_path)
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.qa_model.encode_video(video_tensor)

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']
            row = {
                'video_id': video_sample['video_id'],
                'question': question,
            }
            
            # QA
            if 'choices' in sample:  # CloseQA
                choices = sample['choices']
                if answer is None:  # FIXME: an ugly fix for some benchmarks do not provide GT
                    answer = choices[0]
                if answer in self.choice_letters[:len(choices)]:
                    correct_choice = answer
                else:
                    correct_choice = self.choice_letters[choices.index(answer)]
                qa_results = self.video_close_qa(question, choices, correct_choice)
                row.update({
                    'choices': choices,
                    'answer': answer,
                    'correct_choice': correct_choice,
                    'pred_answer': qa_results['pred_answer'],
                    'pred_choice': qa_results['pred_choice'],
                    'qa_acc': qa_results['acc'] * 100,
                })
                if self.save_choice_scores:
                    row.update({
                        key: qa_results.get(key, '')
                        for key in self.csv_fieldnames
                        if key.startswith('choice_') or key in {
                            'top1_prob',
                            'top2_prob',
                            'prob_margin',
                            'logit_margin',
                            'normalized_choice_entropy',
                        }
                    })
            else:  # OpenQA
                qa_results = self.video_open_qa(question)
                row.update({
                    'answer': answer,
                    'pred_answer': qa_results['pred_answer'],
                })

            if 'question_type' in sample:
                row['task'] = sample['question_type']

            self.append_result(row)


if __name__ == "__main__":
    work(ReKVOfflineVQA)
