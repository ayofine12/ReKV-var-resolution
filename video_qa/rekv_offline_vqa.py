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
        if hasattr(self.qa_model, "multiple_choice_answering"):
            pred_answer = self.qa_model.multiple_choice_answering(
                input_text,
                num_choices=len(candidates),
                retrieved_indices=retrieved_indices,
            )
        else:
            pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16, retrieved_indices=retrieved_indices)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

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
