# MLVU Follow-up Experiments

MLVU `fs112` and `fs224` score CSVs are expected at:

- `/mnt/ssd1/mwnoh/var-resolution-mlvu-confidence/fs112_lb72_rs144/1_0.csv`
- `/mnt/ssd1/mwnoh/var-resolution-mlvu-confidence/fs224_lb18_rs36/1_0.csv`

Use these CSVs only. Do not re-run video QA for verifier experiments unless the
score CSVs are missing or incomplete.

## 1. Sanity Check The Score CSVs

- Confirm both CSVs exist.
- Confirm both contain confidence columns:
  - `top1_prob`
  - `prob_margin`
  - `logit_margin`
  - `choice_entropy`
  - `normalized_choice_entropy`
- Count matched examples between `fs112` and `fs224`.
- Report base accuracy:
  - `fs112` only
  - `fs224` only
- Count disagreement groups:
  - both correct
  - only `fs112` correct
  - only `fs224` correct
  - neither correct

## 2. Analyze Resolution Complementarity

- Analyze where `fs112` and `fs224` differ.
- Break down `only fs112 correct` vs `only fs224 correct` by:
  - task/category
  - question type if available
  - number of choices
  - question prefix such as `what`, `where`, `when`, `why`, `how many`
- Check whether `fs224` is stronger on fine-detail/text-heavy questions.
- Check whether `fs112` recovers some broader-context or spatial/global cases.

## 3. Sweep The fs224 Low-confidence Threshold

Evaluate multiple `fs224.prob_margin` thresholds:

```text
0.10, 0.20, 0.30, 0.40, 0.50
```

For each threshold, compute:

- `fs224 high` count and ratio
- `fs224 low` count and ratio
- `fs112` call ratio
- verifier call ratio
- low-confidence disagreement count
- low-confidence disagreement decisive count
- selective oracle accuracy
- estimated relative cost

Interpretation:

- Lower threshold means cheaper routing but fewer chances to improve.
- Higher threshold means more `fs112` and verifier calls, higher cost, and
  potentially more verifier mistakes.

## 4. Evaluate Confidence-only Routing

For each threshold, evaluate confidence-only routing on low-confidence
disagreement cases.

Main confidence rule:

```text
score = top1_prob + prob_margin - normalized_choice_entropy
```

Choose:

```text
score112 > score224 -> route to fs112
score224 > score112 -> route to fs224
tie -> default fs224
```

Also report simple single-feature baselines:

- higher `top1_prob`
- higher `prob_margin`
- higher `logit_margin`
- lower `normalized_choice_entropy`

Report both:

- diagnostic routing accuracy on low-disagree-decisive examples
- stitched full-set accuracy after applying the route to all examples

## 5. Evaluate LLM Verifier

Run the LLM verifier on the same low-confidence disagreement setting.

Recommended prompt setting:

- Include question and choices.
- Include `fs112` prediction and `fs224` prediction.[]
- Include compact confidence features only:
  - `top1_prob`
  - `prob_margin`
  - `normalized_choice_entropy`
- Do not include gold labels, `qa_acc`, or direct correctness information.
- Do not let the verifier solve the video QA question directly.
- Ask it to choose only between the two candidate predictions.

Recommended first-pass thresholds:

```text
0.20, 0.30, 0.40
```

Run all thresholds only if cost is acceptable.

## 6. Compare Confidence Rule vs LLM Verifier

For each evaluated threshold, compare:

```text
Method                         Accuracy / Routing Accuracy
fs112 only
fs224 only
confidence rule
LLM verifier
selective oracle
full oracle
```

Important comparisons:

- Does LLM verifier outperform the confidence rule?
- Does confidence rule already match LLM verifier?
- Does either method improve full-set accuracy over `fs224` only?
- How much extra cost is required for each gain?

Possible conclusions:

- If LLM verifier > confidence rule:
  - verifier captures useful signals beyond hand-crafted confidence routing.
- If LLM verifier ~= confidence rule:
  - multi-feature confidence routing may be the main source of improvement.
- If LLM verifier < confidence rule:
  - use confidence rule as the primary router and report LLM verifier as a
    weaker or non-essential alternative.

## 7. Prepare Paper-ready Tables

Recommended main table:

```text
Threshold | fs112 Call % | Verifier Call % | fs224 Acc | Confidence Rule Acc | LLM Verifier Acc | Cost
```

Recommended diagnostic table:

```text
Threshold | low-disagree-decisive N | Always fs224 | Always fs112 | Confidence Rule | LLM Verifier | Oracle
```

Recommended analysis table:

```text
Category / Question Type | only fs112 correct | only fs224 correct | fs112 share
```

## 8. Paper Claim Checklist

Before writing the result section, verify:

- The selected threshold was not chosen only from the final test set, or clearly
  describe it as an analysis/ablation if it was.
- Full-set accuracy gain is reported, not only decisive-case routing accuracy.
- Cost increase is reported together with accuracy gain.
- The verifier is described as a meta-router, not as a model that sees video
  frames or re-solves the original VQA task.
- Any category-level claims are supported by enough examples.
