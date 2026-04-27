# ReKV Var-Resolution Conversation Log

Generated: 2026-04-27 UTC

Note: This is a cleaned Markdown transcript of the user-visible conversation. Internal tool logs and low-level execution traces are omitted.

---

## Turn 1

**User**

`/mnt/ssd1/mwnoh/var-resolution-grid/fs56_lb64_rs16/1_0.csv`와 `/mnt/ssd1/mwnoh/var-resolution-grid/fs56_lb64_rs32/1_0.csv`의 accuracy가 너무 낮게 나오는 거 같은데 이유가 뭘까? 일단 `/root/mwnoh/ReKV-var-resolution/tools/run_qwen_frame_block_screen_low_res.sh`랑 `/root/mwnoh/ReKV-var-resolution/tools/run_qwen_frame_block_screen_high_res.sh`의 실험이 다 끝나길 기다려야겠지?

**Assistant**

지금 단계에선 기다릴 필요가 없다고 설명했다. 두 grid 결과는 이미 완료된 결과이고, `rs16`은 약 `27.61`, `rs32`는 약 `29.98`로 4지선다 chance level 근처라고 답했다. 현재 돌고 있는 screen 실험은 grid 결과와 별개이며, `frame_size=56`에서 frame당 token 수가 너무 적고 retrieval block 수도 작아서 긴 LVBench 비디오에서 디테일과 핵심 정보 회수가 약해지는 게 주요 원인이라고 설명했다.

---

## Turn 2

**User**

`/mnt/ssd1/mwnoh/var-resolution-screen` 안에 있는 결과도 분석해줘

**Assistant**

당시 진행 중이던 screen 결과 두 개를 분석했다.

- `fs56_lb32_rs16`: 부분 평균 약 `27.82`
- `fs224_lb32_rs16`: 부분 평균 약 `36.34`

공통 문항 기준으로 `fs224`가 약 `+6.88`p 좋았고, `entity recognition`, `event understanding`, `key information retrieval`, `reasoning` 전반에서 개선이 보인다고 정리했다.

---

## Turn 3

**User**

`fs56`이 더 잘 맞추는 질문들도 있어?

**Assistant**

있지만 소수라고 답했다. 당시 공통 문항 비교에서 `fs56`만 맞춘 문항과 `fs224`만 맞춘 문항을 비교해, `fs56`이 이기는 질문도 분명 있으나 전체적으로는 `fs224`가 우세하다고 설명했다. `event understanding`, `entity recognition` 관련 예시 질문도 일부 제시했다.

---

## Turn 4

**User**

token budget이 한정되어 있을 때, 질문에 따라 낮은 해상도로 많은 frame을 retrieve할지, 높은 해상도로 적은 frame을 retrieve할지 고민 중이라고 설명하고 의견을 물었다.

**Assistant**

아이디어가 유망하다고 평가했다. 다만 현재 실험들은 strict budget-matched가 아니라고 짚었다. `frame_size`에 따라 frame당 visual token 수가 크게 달라서, 겉으로 비슷해 보여도 실제 retrieval token budget이 많이 다르다고 설명했다. 단순 low/high 이분법보다 `low-res scout + selective high-res refine` 같은 coarse-to-fine 구조가 더 현실적이고 구현도 깔끔하다고 제안했다.

---

## Turn 5

**User**

`assert global_h_k.size(-2) <= self.n_init + self.n_local` 에러 traceback을 보여주며 원인을 물었다.

**Assistant**

원인은 `retrieve_size > local_block_count` 조합에서 retrieval로 가져온 KV 길이가 현재 local attention 경로가 수용할 수 있는 범위를 넘어서기 때문이라고 설명했다. 구체적으로 `lb32, rs64` 같은 조합에서 retrieval token 수가 `n_local`보다 커져 assert에 걸린다고 분석했고, 초기 validation과 sweep script 가드로 막는 방향을 우선 제안했다.

---

## Turn 6

**User**

`assert`를 무시하면 문제가 생기냐고 물었다.

**Assistant**

단순히 assert를 지우면 retrieved block 일부가 조용히 잘리는 구조라 의미가 틀어진 실험이 된다고 설명했다. 현재 구현은 retrieved KV를 독립 memory branch로 다루지 않고 `init + local` 구조에 끼워 넣기 때문에, `n_local`보다 긴 retrieved context는 중간 일부가 attention에서 버려진다고 정리했다.

---

## Turn 7

**User**

retrieve 개수와 local window는 원래 독립 아닌가, attention에 들어가는 단계에서 창 밖으로 안 밀려나게 고치면 되지 않냐고 물었다.

**Assistant**

개념적으로는 독립이어야 하지만 현재 코드는 그렇지 않다고 답했다. 이를 해결하려면 `init / retrieved / local-text`를 분리한 3-branch attention 구조가 필요하다고 설명했고, user 요청에 따라 그 구조로 코드를 수정하기로 했다.

---

## Turn 8

**User**

3-branch 구조로 코드를 수정해달라고 요청했다.

**Assistant**

`model/attention/kv_cache_manager.py`와 `model/attention/rekv_attention.py`를 중심으로 3-branch 구조를 구현했다고 답했다. `init`, `retrieved`, `local`을 분리해서 attention에 넣도록 바꿨고, 이전의 `retrieve_size <= local_block_count` 제한도 제거했다고 설명했다. syntax check와 synthetic smoke test를 통과했다고 보고했다.

---

## Turn 9

**User**

실제로 돌려보라고 요청했다.

**Assistant**

`retrieve_size > local_block_count` 조합으로 smoke test를 실제로 돌려봤고, 이전 assert 없이 완료됐다고 보고했다.

- `fs56_lb32_rs64` debug run 성공
- `fs224_lb32_rs64` debug run 성공

single-GPU 환경에서 end-to-end QA loop까지 정상 통과한다고 설명했다.

---

## Turn 10

**User**

`var-resolution-screen` 결과를 분석해달라고 요청했다.

**Assistant**

완료된 screen 결과들을 비교해서 다음과 같이 요약했다.

- best: `fs224_lb32_rs64` 약 `41.81`
- `fs224`가 전반적으로 `fs56`보다 크게 우세
- `retrieve_size`를 키우는 효과는 high-res일 때 더 큼
- `local_block_count`를 늘리는 효과는 거의 없거나 오히려 약간 나쁨

즉 screen setting에서는 큰 local window보다 `high-res + 큰 retrieve_size`가 더 중요하다고 정리했다.

---

## Turn 11

**User**

`fs56`이 `fs224`를 따라가는 질문 유형이 있는지, 그리고 `fs56_rs64` vs `fs224_rs16`, `fs56_rs64` vs `fs224_rs64` 비교를 요청했다.

**Assistant**

다음과 같이 정리했다.

- `event understanding`, `summarization`, 일부 mixed `reasoning`은 `fs56`이 비교적 잘 따라감
- 하지만 `entity recognition`, `key information retrieval`는 `fs224` 우세가 큼
- `rs64`를 충분히 줘도 `fs56`이 `fs224`를 완전히 따라오진 못함

이 결과를 바탕으로 low-res many-frame은 흐름형 질문 후보, high-res는 디테일형 질문 후보라고 정리했다.

---

## Turn 12

**User**

token budget을 고정했을 때 `fs`와 `rs`를 어떻게 맞출지 추천해달라고 요청했다.

**Assistant**

frame당 token 수를 코드에서 직접 계산해 설명했다.

- `fs56 -> 4`
- `fs112 -> 16`
- `fs224 -> 64`

추천 budget으로 `1024`를 먼저 제안했고, 조합은:

- `fs56_rs256`
- `fs112_rs64`
- `fs224_rs16`

이후 tighter budget test로 `256`도 제안했다. 또한 `lb`는 현재 결과상 민감도가 낮아서 우선 `32`로 고정하자고 제안했다.

---

## Turn 13

**User**

`lb=32` 고정으로 `fs56_rs256`과 `fs112_rs64`를 돌리고 싶다고 요청했다.

**Assistant**

전용 실행 스크립트 [run_qwen_frame_block_screen_budget_1024.sh](/root/mwnoh/ReKV-var-resolution/tools/run_qwen_frame_block_screen_budget_1024.sh)를 만들고, 새 실험을 분리해서 돌릴 수 있게 준비했다. 이후 user 요청으로 실행을 중단했고, partial CSV가 남지 않았음을 확인했다.

---

## Turn 14

**User**

`fs56_rs256` 실행 중 `CudaCache.alloc()`에서 `assert len(self.idle_set) > 0` 에러를 보여주었다.

**Assistant**

원인은 `topk=256`인데 GPU-side cache 상한 `max_cached_block`이 `128`로 고정돼 있어서, retrieved block을 올리다 cache slot이 고갈된 것이라고 분석했다. 이후:

- `max_cached_block = max(128, topk)`로 자동 확장되게 수정
- `qwen2_5_vl_rekv.py`에도 그 값이 반영되게 수정

그리고 `fs56_lb32_rs256` smoke test로 실제 retrieval QA 구간까지 통과하는 것을 확인했다.

---

## Turn 15

**User**

`fs56_lb32_rs256`, `fs112_lb32_rs64`, `fs224_lb32_rs16`를 비교 분석해달라고 요청했다.

**Assistant**

`lb32`로 맞춘 공정 비교를 진행했다.

- `fs56_rs256`: `30.67`
- `fs112_rs64`: `36.69`
- `fs224_rs16`: `36.88`

핵심 결론:

- `fs224`: `entity recognition`, `key information retrieval`, `reasoning`
- `fs112`: `event understanding`, 일부 mixed `event + reasoning`
- `fs56`: 몇몇 mixed 예외는 있으나 전체적으로 약함

그래서 adaptive routing 관점에선 `fs56 vs fs224`보다 `fs112 vs fs224`가 더 유망하다고 정리했다.

---

## Turn 16

**User**

`event understanding`, `entity recognition`, `reasoning` 각각의 특성과 예시를 설명해달라고 요청했다.

**Assistant**

세 task를 다음처럼 설명했다.

- `event understanding`: 흐름, 순서, 행동 변화
- `entity recognition`: 사람, 물체, 색, 수, 위치 등 시각 디테일 식별
- `reasoning`: 화면 단서를 조합해 원인/의도/관계/해석을 추론

그리고 실제 CSV에서 각 유형 질문 예시를 제시했다.

---

## Turn 17

**User**

`event understanding`, `reasoning`, `event understanding + reasoning`의 차이와 mixed task에서 `fs112`가 높아 보이는 이유를 물었다.

**Assistant**

다음처럼 설명했다.

- `event understanding`: 사건/행동/순서 자체를 묻는 타입
- `reasoning`: 단서를 조합해 해석하는 타입
- `event understanding + reasoning`: 먼저 사건을 파악한 뒤 그 위에서 해석까지 해야 하는 타입

그리고 mixed task는 표본이 적다고 짚었다.

- `event understanding`: `n=257`
- `reasoning`: `n=68`
- `event understanding + reasoning`: `n=14`

따라서 `fs112` 우세 신호는 있지만 아직 강한 결론은 아니라고 정리했다.

---

## Turn 18

**User**

질문 유형보다 비디오 특성과 더 연관이 있을지, 실제 비디오 분석도 가능한지 물었다.

**Assistant**

먼저 video-level 성향을 정량 분석했고, 이후 실제 프레임 contact sheet까지 만들어 분석했다. 결론은:

- 차이가 특정 비디오에 몰리는 경향이 있음
- source resolution, 편집 방식, 장면 구성 같은 비디오 특성도 중요함
- 하지만 같은 비디오 안에서도 질문에 따라 `fs112`와 `fs224` 우세가 바뀌므로, 비디오만으로 결정하긴 부족함

대표적으로:

- `SRq0weUKskM`: low-res cartoon, fine-grained counting이라 `fs224` 유리
- `EwskdNETNx8`: multi-cut show, temporal flow가 중요해서 `fs112` 유리
- `GcRKREorGSc`: 720p 원본이어도 fine-grained animal recognition은 `fs224`가 유리

즉 `비디오 특성 + 질문 요구`를 함께 보는 2단계 라우팅이 더 적절하다고 설명했다.

---

## Turn 19

**User**

원본 해상도가 낮은 원본에는 `fs224`, 높은 원본에는 `fs112`를 쓰면 되냐고 물었다.

**Assistant**

부분적으로는 맞지만 정확히는 더 미묘하다고 답했다.

당시 partial 결과 기준:

- 원본 `<=360p`: `fs224` 우세
- 원본 `361–720p`: `fs112` 우세
- 원본 `>720p`: 표본은 적지만 당시엔 `fs224` 쪽 신호

그래서 source resolution은 hard rule이 아니라 weak prior로 쓰는 게 낫다고 정리했다.

---

## Turn 20

**User**

`168` 실험을 해볼지 물었다.

**Assistant**

`336`보다 `168`이 더 좋은 다음 실험이라고 답했다. `112`와 `224`가 거의 비기면서도 강점이 갈리기 때문에, 중간 해상도 `168`이 sweet spot일 수 있다고 설명했다. 이후 user 요청으로 [168_32_57.sh](/root/mwnoh/ReKV-var-resolution/tools/168_32_57.sh)를 만들고 실행 가능한 상태로 준비했다.

---

## Turn 21

**User**

`fs112_lb32_rs128`, `fs224_lb32_rs32` 진행 중 결과를 바탕으로 각각 잘 맞추는 질문 유형을 구분해달라고 요청했다.

**Assistant**

partial 결과 기준 공통 문항 비교를 수행했다. 당시엔:

- `fs112_lb32_rs128`: 흐름/맥락/추론 쪽이 강함
- `fs224_lb32_rs32`: 디테일이 섞인 mixed visual task와 summarization 쪽이 상대적으로 강함

다만 mixed task는 표본이 적어 신호 수준으로 보자고 설명했다.

---

## Turn 22

**User**

`fs112_lb32_rs128`, `fs168_lb32_rs57`, `fs224_lb32_rs32` 세 결과를 분석해달라고 요청했다.

**Assistant**

완료된 세 결과를 비교했다.

- `fs112_lb32_rs128`: `37.18`
- `fs168_lb32_rs57`: `39.45`
- `fs224_lb32_rs32`: `38.46`

핵심 결론:

- `fs168`이 현재 가장 좋은 baseline
- `fs112`: `event understanding`, `reasoning`, `summarization`
- `fs224`: `entity recognition + key information retrieval`, `entity recognition + reasoning`
- `fs168`: `entity recognition`, `key information retrieval`, `event understanding + key information retrieval` 등에서 가장 안정적

즉 `168`이 detail과 coverage 사이의 sweet spot처럼 보인다고 정리했다.

---

## Turn 23

**User**

`168`을 베이스라인으로 두고 `112`는 흐름/추론용, `224`는 디테일용으로 쓰자는 해석과, source resolution으로도 fs를 조정할지 물었다.

**Assistant**

그 해석이 현재까지는 가장 자연스럽다고 답했다.

- 기본값: `fs168`
- 흐름/맥락/추론: `fs112`
- 디테일/식별/세부 정보 회수: `fs224`

그리고 source resolution은 보조 prior로 쓰는 게 좋다고 설명했다.

- `<=360p`: `fs224` 쪽 bias
- `480~720p`: `fs112` 쪽 bias
- `1080p+`: 아직 불확실, `fs168` 또는 `fs224`를 기본으로 두는 쪽이 안전

---

## Turn 24

**User**

지금까지 주고받은 대화를 Markdown 파일로 `mwnoh` 아래에 뽑아달라고 요청했다.

**Assistant**

이 파일을 생성했다.

---

## Current Takeaway

- Baseline candidate: `fs168`
- Flow / context / reasoning override: `fs112`
- Fine detail / mixed visual-detail override: `fs224`
- Source video resolution is useful as a prior, but not strong enough to replace question-aware routing.

