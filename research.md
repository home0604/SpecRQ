> **2026-11-05:** 연구 방향이 nested weight-VQ로 전환됨. 현행 계획은 Notion (CLAUDE.md 링크) 참고. 이 문서는 이전 KV-cache GSRQ 계획이며 KV 확장 단계(Notion 04)에서 재사용.

# Format-Aware Asymmetric Residual Quantization for Self-Speculative Decoding

연구의 중심은 **"저정밀 codeword로 확보한 공간을 vector codebook의 표현력에 재투자한다"**입니다. 이 가설을 self-speculative decoding에서 검증합니다. Acceptance-aware 학습과 비대칭 RQ는 이 가설을 실현하는 설계 요소로 둡니다.

제목은 format과 단계별 설계를 드러내도록 정했습니다. Acceptance를 직접 최적화하는 방법은 아직 확정되지 않았으므로, 제목에서 이를 약속하지 않습니다.

---

## 1. 연구 문제와 목표

하나의 RQ 기반 KV 표현을 두 소비자가 사용합니다.

- **Draft:** 첫 단계 또는 짧은 prefix만 읽습니다. 낮은 접근 비용과 충분한 target agreement가 중요합니다.
- **Target:** 전체 단계를 읽습니다. 원본 모델에 대한 충실도가 중요합니다.

제안하는 설계는 **각 단계의 codeword format, codebook 용량, 학습 목적을 함께 선택**합니다.

> **핵심 질문:** 같은 KV index rate와 codebook 저장 공간에서, 저정밀 codeword와 더 풍부한 vector codebook의 조합이 고정밀 codeword 기반 RQ보다 유리한가? 그 이득을 실제 speculative decoding 속도로 연결할 수 있는가?

목표는 codeword MSE를 낮추는 것 자체가 아닙니다. 최종 목표는 target 품질 제약 아래에서 **확정된 출력 token당 실행 시간과 에너지**를 줄이는 것입니다.

---

## 2. 검증할 가설

| 가설 | 내용 | 지지하는 결과 | 반대 결과의 의미 |
|---|---|---|---|
| **H1. Format 적합성** | 같은 bit 수라도 codeword에 맞는 format은 단계, K/V, 학습 목적에 따라 다르다. | 동일 구조에서 format별 품질 차이가 재현됨 | format 선택보다 scale·학습 방식이 중요할 수 있음 |
| **H2. 정밀도–표현력 교환** | codeword 정밀도를 낮추고 \(d, K\)를 늘리면 같은 자원에서 draft 품질이 더 높다. | 동일 index rate·codebook 용량에서 acceptance 개선 | 저정밀화 손실이나 고차원 학습 난도가 이득을 상쇄 |
| **H3. 비대칭 단계 구성** | draft prefix와 refinement에 서로 다른 format·용량을 배정하는 편이 균일 구성보다 낫다. | 동일 총자원·target 품질에서 acceptance–latency 관계가 더 좋음 | 대칭 구성으로 충분하거나 단계별 관리 비용이 큼 |
| **H4. 목적함수 적합성** | query/attention 또는 agreement를 고려한 학습이 저정밀 prefix의 유용성을 높인다. | 같은 format·구조에서 실제 accepted length 개선 | local distortion 개선이 출력 분포까지 전달되지 않음 |
| **H5. 시스템 이득** | 위 품질 이득과 저장 공간 절감이 복원·검색 비용을 포함해도 유효하다. | end-to-end token/s 또는 energy/token 개선 | 알고리즘 이득은 있으나 실행 구조 재설계가 필요 |

**역할 분담:** H2가 중심 가설입니다. H1은 원인 분석, H3·H4는 설계 방법, H5는 시스템 검증입니다.

"첫 단계는 크고 저정밀, 뒤 단계는 작고 고정밀"은 H3의 유력한 후보로 두되, 정답으로 고정하지 않습니다.

---

## 3. 표현 구조와 자원 정의

단계 \(s\)의 codebook을 \(C_s\), format을 \(f_s\), index를 \(z_s\)라 하면

\[
\hat{x}^{(m)}=\sum_{s=1}^{m}Q_{f_s}(C_s[z_s]).
\]

Draft는 \(m\)단계, target은 \(S\)단계를 씁니다. 처음에는 \(m=1\)로 고정하고, 이후 prefix 길이까지 확장합니다.

| 변수 | 의미 |
|---|---|
| \(d_s\) | sub-vector 차원 |
| \(K_s\) | codeword 수 |
| \(b_s, f_s\) | codeword 원소의 bit 수와 format |
| scale 공유 단위 | codebook 전체 / group / codeword별 scale |
| codebook 공유 단위 | layer, KV head, sub-vector 위치 사이의 공유 여부 |
| 학습 목적 | L2, query-weighted, attention-output, logit agreement |
| 실행 경로 | gather–decode 또는 lookup 기반 계산 |

대략적인 비용은 다음과 같습니다.

\[
r_{\mathrm{index},s}=\frac{\lceil\log_2K_s\rceil}{d_s},\qquad
M_{\mathrm{CB},s}=\frac{K_sd_sb_s}{8}+M_{\mathrm{metadata},s}
\]

실제 비용에는 codebook 개수, padding, scale, transform 저장 비용을 더합니다.

**codeword 저장 bit, KV index bit, 연산 precision은 반드시 분리해서 기록합니다.** 예를 들어 INT4 codeword를 FP16으로 복원해 계산하는 구성도 별도의 후보입니다.

---

## 4. Format 확인 범위

| 실험군 | 목적 |
|---|---|
| FP16 | 고정밀 기준점 |
| INT8, INT4 | 단순하고 규칙적인 표현의 기준점 |
| NF4 | 비균일 scalar level의 효과 |
| Posit4, Posit8 | 다른 동적 범위·간격 배분의 효과 |
| 학습형 4bit scalar level | 고정 format의 한계를 판단하는 비교군 |

재현 가능한 비교를 위해 Posit은 exponent 설정(es)을, NF4는 scaling 규칙을 명시합니다.

학습형 scalar level은 **현실적인 공유 범위와 table 비용을 포함한 비교군**으로 둡니다. 결과는 다음과 같이 해석합니다.

- 학습형 level만 유리하다 → 고정 format보다 분포 적응이 중요하다.
- NF4/posit도 비슷하게 유리하다 → 저비트 비균일 표현이 실용적이다.
- INT4도 재학습 후 따라온다 → format 자체보다 format-aware codebook 학습이 핵심이다.

Gain–shape 분리 저장은 별도 실험으로 둡니다. format, gain precision, scale 단위를 처음부터 함께 바꾸면 원인을 분리하기 어렵습니다.

---

## 5. 실험 단계

| 단계 | 실험 | 고정할 조건 | 주요 결과 |
|---|---|---|---|
| **E0. 실행 가능성** | 기존 RQ를 KV에 적용하고 draft–verify reference 구현 | weight precision, decoding policy | 실제 acceptance와 실행 시간을 측정할 기반 |
| **E1. 순수 format 효과** | 같은 codebook을 여러 format으로 저장·복원 | \(d, K\), scale 정책, codebook 공유 | format별 손실, 민감한 layer와 K/V |
| **E2. Format-aware 학습** | 사후 변환 vs 복원값을 반영한 재학습·재할당 | 구조와 저장 예산 | 학습으로 회복되는 손실 |
| **E3. 정밀도–표현력 교환** | 저정밀 대형 vs 고정밀 소형 codebook | index rate, codebook 용량 | H2 직접 검증 |
| **E4. 단계별 비대칭 설계** | uniform vs format-only vs capacity-only vs 둘 다 | 전체 index rate, 총 codebook 용량 | 비대칭 구성의 기여 |
| **E5. Draft 목적 학습** | L2 vs query-weighted vs (필요시) logit loss | format과 구조 | H4, target nesting 비용 |
| **E6. 시스템 평가** | 실제 저장·복원·encoding·draft–verify 실행 | target 품질 기준 | 속도, 메모리, 에너지 |

모든 구성을 처음부터 full-model rollout으로 평가하지 않습니다. **local replay로 선별 → 모델 forward → shortlist rollout** 순서로 비용을 줄입니다.

다만 E0에서 작은 rollout 환경은 먼저 만들어 둡니다. 그래야 이후 실험이 MSE 개선만 좇는 방향으로 흐르지 않습니다.

---

## 6. H2 검증을 위한 비교 예

**index rate 1 bit/element, codebook 하나당 약 4 KiB**를 맞춘 후보입니다(metadata 제외).

| 구성 | \(d\) | \(K\) | 원소 precision | Index rate | Codebook |
|---|---:|---:|---:|---:|---:|
| 고정밀 소형 | 8 | 256 | FP16 | 1 bit | 4 KiB |
| 저정밀 확대 | 10 | 1024 | 3 bit | 1 bit | 3.75 KiB |

- 4bit format에서 정확히 같은 용량의 \((d, K)\)가 나오지 않으면, 억지로 맞추지 않고 **여러 용량 예산에서 feasible configuration의 Pareto frontier**를 비교합니다.
- \(d\)가 head dimension을 나누지 못하면 padding이나 불균일 partition이 필요합니다. 그 비용도 실제 rate에 포함합니다.
- **같은 \(K\)에서 \(d\)만 늘리는 실험은 따로 수행합니다.** 이 실험은 같은 rate에서 품질을 높이는 실험이 아니라, index rate를 낮추면서 품질이 얼마나 유지되는지 보는 실험입니다.

---

## 7. 실험 환경

GPU reference 구현과 하드웨어 비용 분석은 분리합니다.

| 항목 | 초기 환경 | 확장 환경 |
|---|---|---|
| 모델 | long-context MHA 모델 1개 | GQA 모델 1개 이상 |
| Weight | draft/target 같은 precision | 실제 저정밀 weight draft와 결합 |
| Context | 4K, 16K, 32K | 모델이 지원하는 더 긴 context |
| Batch | 1, 8, 32 | 메모리 여유와 병목에 따라 확대 |
| Draft length | 4에서 시작해 고정 | 2, 4, 8, 16 sweep |
| Decoding | greedy와 sampling 구분 | temperature 변화 |
| Workload | 일반 텍스트, 장문 문서 | 코드, 추론, 장거리 정보 활용 |
| 장치 | GPU 한 종류에서 통제 실험 | 대역폭/compute 비율이 다른 GPU 또는 가속기 모델 |

MHA는 이득이 드러나는 조건이고, GQA는 적용 범위와 한계를 확인하는 조건입니다. GQA를 빼기보다 **KV traffic 비중이 어느 정도부터 이득이 생기는지**를 보여줍니다.

- calibration, configuration selection, 최종 평가는 문서 단위로 분리합니다.
- query metric을 쓸 때는 평가 문서의 미래 query로 codebook을 학습하지 않습니다.
- RoPE 적용 전후 중 어느 표현을 양자화하는지 모든 실험에서 명시합니다. 현재 VQLLM은 pre-RoPE입니다.

---

## 8. 측정 지표와 baseline

| 층 | 지표 |
|---|---|
| 표현 | 실제 bits/element, codebook·metadata 크기, score/output 오차 |
| 생성 품질 | TV, greedy agreement, 평균 accepted prefix length, target 품질 |
| 실행 | draft/verify/encoding 시간, 확정 token/s, peak memory, energy/token |

KV encoding 비용에는 **prefill의 일괄 encoding과 decode 중 신규 token encoding**을 모두 포함합니다. 큰 \(K\)의 최근접 탐색은 이 연구의 주요 비용입니다. 탐색 시간을 뺀 결과만으로 시스템 이득을 주장하지 않습니다.

필수 baseline:
- 일반 고정밀 codeword RQ
- 각 format에 맞춘 **대칭 RQ**
- GSRQ
- 계층형 scalar quantization
- 같은 총 rate에서 독립적으로 최적화한 target RQ → **nesting으로 잃는 품질** 확인
- 같은 target의 비투기적 autoregressive decoding → **speculation 자체의 실제 이득** 확인

---

## 9. 가장 먼저 할 실험과 진행 판단

첫 판단은 대규모 탐색보다 다음 세 질문에 집중합니다.

1. **1 bit/element 안팎의 RQ prefix에서 codeword를 FP16에서 INT4/NF4/posit4로 바꾸면 손실이 얼마나 큰가?**
2. **format-aware 재학습으로 그 손실을 회복할 수 있는가?**
3. **같은 저장 예산에서 codebook을 키우면 복원 오차뿐 아니라 accepted length도 좋아지는가?**

결과에 따라 연구의 무게중심을 정합니다.

- **format 차이가 크고 재현된다** → format 설계와 그 이유를 중심 기여로
- **format 차이는 작지만 저정밀 대형 codebook이 유리하다** → precision–capacity allocation을 중심 기여로
- **같은 자원에서 품질은 좋아지지만 실행이 느리다** → encoding·gather를 해결하는 아키텍처가 핵심 기여
- **품질 이득이 없다** → 큰 codebook보다 sharing, scale, gain–shape 구조를 재검토

논문의 중심 주장은 "특정 format이 더 좋다"가 아닙니다. **"vector codeword의 scalar precision은 고정된 상수가 아니라, codebook 표현력·실행 비용과 함께 배분해야 할 설계 자원이다"**입니다. Self-speculative decoding의 두 소비자가 이 주장을 검증하고, 비대칭 설계로 발전시키는 근거가 됩니다.

---

## 10. 현재 상태 (2026-10-20)

### 환경
- VQLLM을 `SpecRQ/VQLLM`에 새로 clone했고 conda env는 `VQLLM_CUDA`입니다. 경로와 주의사항은 `CLAUDE.md`에 정리했습니다.
- **학습된 codebook이 없습니다.** 이전 Llama-3.2-3B codebook은 삭제됐습니다. 현재 clone에는 `llama3_3b` builder가 없고, 지원 모델은 llama3_8b(+instruct), llama2_7b, mistral_7b, gemma_7b입니다.
- `run_rqllm_experiment.sh`의 `CKPT_ROOT_DIR`(`/data/home/ssk/ckpts`)은 존재하지 않는 경로입니다. 학습 전에 설정해야 합니다.

### 이전 탐색에서 확인한 사항 (Llama-3.2-3B, 삭제된 코드 기준이라 재현이 필요함)
- **Key layout 버그:** HF `k_proj` 출력을 Meta layout으로 되돌리지 않고 codebook을 적용하면, full-stage key 상대오차가 1.3%가 아니라 44%로 나옵니다. 이전 acceptance/TVD 실험 결과는 이 버그의 영향을 받았을 수 있습니다.
- **E1 예비 결과**
  - codebook: GSRQ, \(S=16\), \(K=256\), \(d=16\) → stage당 0.5 bit/element
  - 평가: WikiText-2 test 2048-token window 8개, teacher-forced, 모든 KV 위치 양자화
  - α, greedy, acc@γ=4는 FP16 codebook을 쓴 전체 16 stage target과 비교한 값입니다.
  - target(16 stage, FP16 codeword)의 PPL은 7.513으로, 원본 모델(7.515)과 사실상 같습니다.
  - **재할당(requant)만으로 저비트 손실을 거의 회복합니다.** INT4/FP4 codeword의 target PPL은 index를 FP16 codebook으로 고르면(fp16idx) 7.77–7.88이고, 변환된 codebook으로 다시 검색하면 7.52입니다.
  - **FP4 + codeword별 scale + 재할당 기준 draft 품질**

    | m (bit/elem) | PPL | α | greedy | acc@γ=4 | K err | V err |
    |---|---:|---:|---:|---:|---:|---:|
    | 1 (0.5) | 67.2 | 0.33 | 0.39 | 0.63 | 0.37 | 0.68 |
    | 2 (1.0) | 16.7 | 0.60 | 0.63 | 1.47 | 0.28 | 0.53 |
    | 4 (2.0) | 8.59 | 0.84 | 0.82 | 2.54 | 0.17 | 0.32 |

  - **현재 RQ prefix는 draft로 쓰기 어렵습니다.** RQ prefix는 원래 target을 위해 학습된 것이라 1 bit 이하의 draft로는 부족합니다. 이는 H3/H5의 동기가 됩니다(draft용 첫 stage를 따로 키우거나 학습하는 방향).
  - **V가 K보다 훨씬 어렵습니다.** m=2에서 V 오차는 0.53, K 오차는 0.28이므로, K/V 비대칭 배분도 후보입니다.
  - posit/fp8 결과는 scale search 범위 문제로 무효였고, 고친 뒤의 재측정은 완료하지 못했습니다.

### QuantSpec 재현 (2026-10-20, `specdec/`, `results/specdec/quantspec_repro/`)
- 설정: Llama-2-7B-32K-Instruct, PG-19 test 앞부분 prompt 10개, greedy, 90 token 생성, batch 1.
  - QuantSpec: KV group 128, FP 잔차 256. 논문에 weight 양자화 방법이 없어 **draft weight는 RTN-W4 g128**로 둠.
  - sparse baseline의 budget은 ctx/4.
- 수락률 = 수락 수 / 제안 수. QuantSpec의 γ별 감소 곡선과 일치하는 정의.
- **8K (Table 6a):**

  | 방법 | γ | 우리 | 논문 |
  |---|---|---:|---:|
  | QuantSpec W4 | 1/2/4/6 | 93.8/89.2/84.3/76.3 | 91.9/89.9/83.2/77.1 |
  | StreamingLLM | 1/2/3 | 91.7/89.1/80.7 | 90.8/89.4/90.2 |
  | SnapKV | 1/2/3 | 93.2/88.9/83.3 | 94.4/91.0/91.8 |

  - QuantSpec은 모든 γ에서 ±2%p 안. sparse는 γ≤2에서 ±2%p 안.
  - 논문의 sparse γ=3 값은 γ=2보다 높아 정의상 모순이고, 같은 논문 γ=1 값에서 예측하면 82.6. 비교에서 제외.
- **4K (Table 3, 최적 γ):** StreamingLLM 87.3 (논문 88.9), SnapKV 91.3 (93.6), QuantSpec 92.2 (92.5).
- **weight 효과 분리 (8K):** QuantSpec KV만 양자화(W16)한 결과는 γ=1/2/4/6에서 96.5/95.1/92.0/89.9.
  - W4를 더하면 2.7/5.9/7.7/13.6%p 하락. γ가 클수록 weight 손실이 누적됨.
  - 따라서 W까지 중첩 표현으로 확장할 동기가 있음 (선행연구: SPEQ, Any-Precision LLM, MatQuant, VPTQ).

### 중첩 weight 1단계 (2026-10-21, `results/specdec/nested_weight/`)
- 방식: weight를 QuantSpec식 계층 INT8(g128)로 저장. draft는 상위 4bit, target은 INT8을 읽음. 별도 draft 사본 없음.
- 8K PG-19, 수락률 % (괄호는 tokens/iter):

  | 구성 | γ=1 | γ=2 | γ=4 | γ=6 |
  |---|---:|---:|---:|---:|
  | KV 계층만 (W16) | 96.5 | 95.1 | 92.0 | 89.9 |
  | W RTN4 사본만 | 94.0 | 89.5 | 83.4 | 79.5 |
  | W 중첩8만 | 91.9 | 89.8 | 85.5 | 79.1 |
  | QuantSpec (W RTN4 + KV 계층) | 93.8 | 89.2 | 84.3 | 76.3 |
  | W 중첩8 + KV 계층 | 93.3 | 90.5 | 84.2 | 77.6 |

- Target PPL (8K, prompt 10개): FP16 8.434, W8 8.459, KV 계층 8.439, W8 + KV 계층 8.461. W8 비용은 +0.3%.
- 해석:
  - scalar bit-plane 중첩 weight는 별도 RTN4 사본과 수락률이 같음 (prompt 10개 기준 차이 ±2%p로 noise 수준).
  - 따라서 draft 사본 3.4GB를 없애는 대가는 target PPL +0.3%뿐.
  - **이 W·KV 동시 bit-plane 중첩이 우리 RQ가 넘어야 할 baseline.**
  - prompt 10개의 noise가 ±1.5–2%p라, 최종 비교는 prompt 50개 이상으로.

### 기준선 확정: prompt 50개 (2026-10-21, `results/specdec/baseline50/`, `specdec/sweep_baseline50.sh`)
- 설정은 위와 같고 prompt만 50개. 수락률 %, 괄호는 prompt 간 표준오차.

  | 구성 | γ=1 | γ=2 | γ=3 | γ=4 | γ=6 |
  |---|---:|---:|---:|---:|---:|
  | KV 계층만 (`qs_w16`) | 97.2 (0.4) | 95.5 (0.6) | | 92.5 (0.8) | 89.3 (1.0) |
  | W RTN4 사본만 (`w4_kvfp`) | 95.6 (0.4) | 91.8 (0.7) | | 86.7 (1.0) | 83.3 (1.3) |
  | W 중첩8만 (`n8_kvfp`) | 94.5 (0.6) | 91.5 (0.8) | | 87.4 (0.8) | 82.5 (1.4) |
  | QuantSpec (`qs_w4`) | 94.4 (0.6) | 90.8 (0.8) | | 85.7 (1.2) | 80.4 (1.5) |
  | W 중첩8 + KV 계층 (`n8_kvqs`) | 94.0 (0.5) | 92.0 (0.7) | | 86.2 (0.9) | 80.9 (1.3) |
  | StreamingLLM (`stream`, W16) | 91.7 (0.8) | 88.2 (0.9) | 83.5 (1.2) | | |
  | SnapKV (`snapkv`, W16) | 93.4 (0.6) | 87.9 (0.7) | 82.7 (0.9) | | |

- 같은 prompt끼리 짝지은 차이 (%p, ± 표준오차), γ=1/2/4/6:
  - 중첩8 − RTN4 사본 (KV 정확): −1.1±0.7 / −0.3±0.9 / +0.7±1.1 / −0.8±1.7 → **차이 없음**.
  - 중첩8+KV계층 − QuantSpec: −0.4±0.7 / +1.2±1.0 / +0.4±1.4 / +0.5±1.6 → **차이 없음**.
  - KV계층만 − QuantSpec: +2.8 / +4.6 / +6.8 / +8.9 (모두 유의) → 수락률 손실의 대부분은 W4 draft weight에서 옴.
- QuantSpec 논문 값(91.9/89.9/83.2/77.1)과는 γ=1,2에서 +1~2.5%p, γ=4,6에서 +2.5~3.3%p 차이. 방향은 같고 다소 높음.
- 결론: 10-prompt 때의 해석이 그대로 유지됨. scalar bit-plane 중첩(W·KV)이 별도 사본과 수락률이 같으므로, 이것이 RQ의 기준선.
  RQ가 보여야 할 것은 같은 저장 bit에서 (a) draft 수락률 상승 또는 (b) target PPL 손실 감소.
- 주의: 위 KV 계층 결과는 모두 `--target full --kv-scale paper`(당시 기본값)이고, n8 weight도 QuantSpec 식 scale.

### 저bit 계층 (2026-10-21 진행 중, `results/specdec/lowbit/`, `specdec/sweep_lowbit.sh`)
- 측정 원칙 (사용자 결정): **수락률 측정 시 target은 원본 그대로**(`--target fp16`, 기본값). 초반 stage/상위 bit 근사의 손실은
  draft에만 반영. "근사 + 잔차로 target이 얼마나 복원되는가"는 2단계 실험(`--target full`)으로 따로 본다.
- QuantSpec scale 공식의 저bit 문제: \(S_U = 2^{b_l}(\max-\min)/(2^{b_u+b_l}-1)\), lower ∈ [−2^{b_l−1}, 2^{b_l−1}−1].
  4+4에서는 무해하지만 2+2에서는 상위 code가 범위 상단 20%에 도달하지 못하고, lower가 [−S_U/2, S_U/4]로 잘려 최댓값(outlier)이 깎인다.
  그 결과 W 2+2 target PPL이 1.5×10⁵로 붕괴. 이 공식으로 돌린 결과는 `lowbit/_paper_scale/`에 보관.
- 수정 (`--kv-scale rtn`, 기본값; weight `nested_int`도 동일): 상위 = 전체 범위 RTN(\(S_U = (\max-\min)/(2^{b_u}-1)\)), 하위 = 상위 cell 안의
  mid-rise \(2^{b_l}\) 단계. draft는 별도 RTN b-bit와 정확히 같아지고, target 오차는 2+2에서 12.8% (평면 RTN4 10.0%; 무작위 행렬 기준).
- 참고: 가우시안 데이터에서는 clipping이 있는 paper 상위 2bit가 MSE는 더 낮음(42% vs 50%). 공정한 scalar 기준선에는 clip search가 필요할 수 있음.
- 결과 (8K PG-19, prompt 50개, 수락률 %, 괄호 표준오차):

  1단계 — draft만 저bit, target 원본

  | draft | γ=1 | γ=2 | γ=4 |
  |---|---:|---:|---:|
  | KV 3bit (`kv3_tfp`) | 96.7 (0.4) | 95.4 (0.5) | 91.5 (0.8) |
  | KV 2bit (`kv2_tfp`) | 92.2 (0.6) | 87.2 (0.8) | 78.5 (1.2) |
  | W RTN 3bit (`w3_kvfp`) | 89.2 (0.7) | 83.2 (0.9) | 75.9 (1.1) |
  | W RTN 2bit (`w2_kvfp`) | 68.0 (1.2) | 55.5 (1.6) | 40.1 (1.6) |
  | W 2bit + KV 2bit (`w2_kv2tfp`) | 59.6 (1.4) | 47.8 (1.5) | 32.1 (1.5) |

  (4bit 참고: KV `qs_w16` 97.2/95.5/92.5, W `w4_kvfp` 95.6/91.8/86.7; baseline50)

  2단계 — target = 상위 + 하위 복원

  | 구성 | target bit | target PPL (FP16 8.434) | γ=1 | γ=2 | γ=4 |
  |---|---:|---:|---:|---:|---:|
  | KV 3+3 (`kv3_full`) | 6 | 8.437 | 96.6 | 94.5 | 92.4 |
  | KV 2+2 (`kv2_full`) | 4 | 8.482 | 90.1 | 84.2 | 74.5 |
  | W 3+3 (`n6_kvfp`) | 6 | 8.467 | 88.8 | 83.2 | 75.2 |
  | W 2+2 (`n4_kvfp`) | 4 | 9.516 | 69.4 | 57.1 | 40.4 |
  | W 2+2 + KV 2+2 (`n4_kv2_full`) | 4 | 9.605 | 62.3 | 49.3 | 32.5 |

- 해석:
  - KV는 scalar 3bit draft까지 거의 무손실, 2bit에서 처음 92%. KV RQ의 승부처는 2bit 이하(m ≤ 4).
  - weight는 3bit부터 떨어지고 2bit에서 scalar draft가 무너짐(γ=1 68%). **weight RQ 2bit draft가 가장 큰 개선 여지.**
  - 6bit target 복원은 KV·W 모두 거의 무손실. 4bit target은 W가 PPL +12.8%로 비쌈 → 잔차 복원 실험의 지점은 W 2+2.
  - KV 2+2는 target을 복원해도 수락률이 1단계보다 2~4%p 낮음(target 오차와 draft 오차가 서로 다른 방향).

### KV RQ 비교 준비 (2026-10-21)
- 코드북 학습 중: GPU 0 `K16C256D16`(0.5 bit/stage, 16 stage = 8bit), GPU 1 `K16C256D32`(0.25 bit/stage, 16 stage = 4bit).
  같은 2bit draft = D16 m=4 / D32 m=8, 4bit target = D16 8 stage / D32 16 stage.
- `RQRep` 공정성 보완: `--rq-window 128`(기본값)으로 QuantSpec과 같은 최근 128–256 토큰 원본 유지 (오래된 block은 post-RoPE K를
  역회전 → 인코딩 → 재회전), `--target-stages`로 target이 읽는 stage 수 지정. 무작위 코드북으로 plumbing 확인
  (RoPE 왕복 오차 5e-7, target fp16일 때 target cache 정확히 원본).
- bit 표기: scalar 계층 2bit는 그룹 scale/zero 포함 2.25bit, RQ는 정확히 2.0bit (코드북은 고정 크기).
- weight RQ는 보류 (사용자 결정).

### KV RQ D16 결과 (2026-10-22, `results/specdec/rq/`, `specdec/sweep_rq.sh d16`)
- 코드북: `K16C256D16` SubVec1, VQLLM fast mode, wikitext 학습. 평가는 PG-19 8K, prompt 50개, greedy, recent window 128.
- 실제 KV 재구성 오차 (L15, 상대오차): 2bit RQ key 0.168 / value 0.336 vs scalar 0.246 / 0.513 (RQ 우세),
  3bit 비슷, 4bit 이상은 scalar 우세 (RQ는 stage당 오차 ×0.78, scalar는 0.5bit당 ×0.71).
- 1단계 (target 원본), 수락률 % (괄호 표준오차), 같은 bit scalar와 짝지은 차이:

  | draft | γ=1 | γ=2 | γ=4 | scalar 대비 (γ=1/2/4) |
  |---|---:|---:|---:|---|
  | RQ 2bit (m=4) | 90.8 (0.8) | 85.1 (1.1) | 78.6 (1.6) | −1.4 / −2.1 / +0.1 (`kv2_tfp`) |
  | RQ 3bit (m=6) | 93.2 (0.6) | 89.4 (0.8) | 84.4 (1.2) | −3.5 / −6.0 / −7.1 (`kv3_tfp`) |
  | RQ 4bit (m=8) | 94.9 (0.6) | 92.0 (0.7) | 88.3 (1.0) | −3.2 / −4.9 / −7.6 (`kv4_tfp`) |
  | scalar 4bit (`kv4_tfp`) | 98.0 (0.3) | 97.0 (0.4) | 95.9 (0.5) | – |

- 2단계 (target = RQ prefix + 잔차): 2bit draft + 4bit target(t=8) 94.1 / 90.7 / 84.5, scalar 2+2(`kv2_full`) 대비
  **+4.1 / +6.5 / +10.0 %p**. target PPL: RQ 4bit 8.513, 8bit 8.458 vs scalar 4bit 8.482, 8bit 8.439 (FP16 8.434).
- 해석:
  - draft만 놓고 보면(target 원본) RQ는 같은 bit scalar보다 나쁨. 2bit에서 MSE는 RQ가 훨씬 낮은데도 동률 → MSE가 attention 충실도를
    대변하지 못함 (scalar key의 채널별 scale이 outlier 채널을 보존하는 효과로 추정, 확인 필요).
  - 중첩 구조의 이득은 2단계에서 나타남: draft가 target의 prefix이므로 두 쪽 오차가 같은 방향이라 일치율이 높음.
    scalar 계층은 target을 복원하면 수락률이 내려가지만(92.2→90.1) RQ는 올라감(90.8→94.1). 대가는 target PPL +0.9%.
  - 코드북 비용 미반영: SubVec1 D16은 stage당 layer당 4.25 MiB(K+V), draft m=4 전 layer 544 MiB로 L2(72 MB)에 안 들어감.
    sub-vector 공유 코드북이 필요 (→ 다음 학습 후보).

### KV RQ D32 결과 (2026-10-23, `results/specdec/rq/`, `specdec/sweep_rq.sh d32`)
- 코드북: `K16C256D32` SubVec1 (stage당 0.25bit). 평가 조건은 D16과 같음. 차이는 prompt별 짝지은 평균 (±표준오차, %p).
- 1단계 (target 원본), 수락률 % γ=1/2/4:

  | draft | D32 | D16 대비 | scalar 대비 |
  |---|---|---|---|
  | 2bit (m=8) | 91.9 / 86.2 / 79.3 | +1.1±0.4 / +1.2±0.5 / +0.8±0.8 | −0.4±0.8 / −1.0±1.0 / +0.9±1.4 |
  | 3bit (m=12) | 94.9 / 91.0 / 87.5 | +1.8±0.5 / +1.6±0.6 / +3.1±0.9 | −1.8±0.6 / −4.4±0.8 / −4.0±1.1 |

- 2단계: 2bit draft + 4bit target(t=16) 95.2 / 92.0 / 87.3. scalar 2+2 대비 **+5.1 / +7.8 / +12.8 %p**,
  D16 m4/t8 대비 +1.1 / +1.3 / +2.8. target PPL 8.487 (D16 4bit 8.513, scalar 4bit 8.482, FP16 8.434).
- 해석: 같은 bit에서 차원을 키우면 draft와 target이 모두 좋아짐 (H2 방향과 일치). 2bit draft는 scalar와 통계적으로 동률이 되고,
  4bit target PPL도 scalar 수준까지 옴. 3bit 이상 draft는 여전히 scalar가 우세. 코드북 크기는 D와 무관하게 stage당 layer당
  약 4.1 MiB라(gain만 D에 반비례), D32 m=8 draft는 전 layer 약 1.05 GiB로 L2 문제가 D16보다 더 큼.

### target 충실도 곡선 (2026-10-23, `results/specdec/target_curve/`, `specdec/sweep_target_curve.sh`)
- `specdec.ppl --ref`: 같은 토큰을 원본(bf16 weight, FP16 KV)과 target에 teacher-forcing으로 넣어 위치별
  KL(원본‖target)과 top-1 일치율을 측정. top-1 일치율 = target을 원본의 γ=1 draft로 썼을 때의 수락률. PG-19 8K, prompt 10개.

  | target | bit | PPL | KL (nats/token) | top-1 일치 % |
  |---|---:|---:|---:|---:|
  | D16 t4 / t6 / t8 | 2 / 3 / 4 | 8.719 / 8.577 / 8.513 | 0.0341 / 0.0164 / 0.0095 | 93.7 / 95.7 / 96.7 |
  | D16 t10 / t12 / t16 | 5 / 6 / 8 | 8.489 / 8.474 / 8.458 | 0.0064 / 0.0048 / 0.0028 | 97.4 / 97.9 / 98.4 |
  | D32 t8 / t12 / t16 | 2 / 3 / 4 | 8.655 / 8.535 / 8.487 | 0.0270 / 0.0119 / 0.0065 | 94.4 / 96.3 / 97.3 |
  | scalar 2+2 / 3+3 / 4+4 | 4 / 6 / 8 | 8.482 / 8.437 / 8.434 | 0.0111 / 0.0010 / 0.0006 | 95.6 / 98.6 / 99.1 |
  | FP16 | 16 | 8.434 | 0 | 100 |

- 4bit에서 PPL은 scalar 2+2(8.482) ≈ D32(8.487)이지만 KL은 D32가 0.0065로 scalar(0.0111)의 0.6배, top-1은 +1.6%p.
  PPL만으로는 분포 차이가 가려짐. 6bit 이상은 scalar가 크게 우세(RQ는 stage를 늘려도 수렴이 느림).

### head 공유 코드북 SV8 (2026-10-23, `results/specdec/rq/sv8_*`)
- D16, SubVec8(한 head의 8개 sub-vector가 shape 공유, gain은 sub-vector별). 코드북 1/8 → draft m=4 전 layer 약 68 MiB(fp16), L2에 들어감.
- SV1 대비 짝지은 차이 (γ=1/2/4, %p): m4 +0.1 / −0.2 / +0.5, m6 +0.1 / −0.2 / −0.3, m8 +0.2 / −0.2 / −1.0, m4/t8 −0.2 / −0.3 / +0.4
  (표준오차 0.4~1.3). target PPL / KL: t8 8.519 / 0.0099 (SV1 8.513 / 0.0095), t16 8.453 / 0.0022 (SV1 8.458 / 0.0028).
- 공유해도 손실 없음. 공유 코드북에서는 sub-vector별 gain이 0인 codeword가 많음(layer 0 key stage 0은 256개 중 평균 33개 사용).
- D32 SubVec4, D128, D64 학습은 2026-10-24 01:34–02:44 사이에 traceback 없이 모두 중단됐습니다(SV4 stack_iter 568/1024, D128 363/1024, D64 263/1024).
  checkpoint는 없습니다. 사용자 결정으로 학습은 보류 중입니다(2026-11-04). 로그: `/data/lsh/logs/vqllm/l2_32k_K16C256{D32_SV4,D128,D64}.log`.

### 양자화 target 기준 (2026-10-23, `results/specdec/t4/`, `specdec/sweep_t4.sh`)
- 주 설정 변경: target ≈ 4 BPA (long context에서는 KV 양자화가 사실상 필수라 FP16 target은 비현실적). 3bit target도 측정.
- scalar 비교군 `--target sep{b}`: 일반 b-bit RTN target + 별도 저bit RTN draft 사본 (중첩 아님, 저장 = draft + target bit).
- 수락률 % γ=1/2/4 (target 4bit):

  | draft | D32 RQ 중첩 (저장 4b) | scalar 별도 사본 (저장 4b + draft) | RQ − scalar |
  |---|---|---|---|
  | 1bit | 89.0 / 84.3 / 76.5 | 7.4 / 4.2 / 2.1 (1bit RTN은 붕괴) | – |
  | 1.5bit | 92.4 / 88.8 / 82.1 | – | scalar 2bit 대비 +0.4 / +0.9 / +2.6 |
  | 2bit | 95.2 / 92.0 / 87.3 | 92.0 / 87.9 / 79.5 | +3.2±0.7 / +4.1±1.0 / +7.8±1.4 |
  | 2.5bit | 96.3 / 94.0 / 89.9 | – | |
  | 3bit | 97.6 / 96.0 / 94.1 | 96.9 / 95.6 / 91.9 | +0.6 / +0.5 / +2.2 |

  중첩 scalar 2+2: 90.1 / 84.2 / 74.5.
- target 3bit: D32 m4 / m6 / m8 = 90.6 / 86.9 / 78.5, 94.5 / 91.7 / 85.0, 96.8 / 94.8 / 90.2.
  scalar 2bit + 3bit target 91.5 / 86.3 / 80.1 → RQ m8 +5.4 / +8.5 / +10.1.
- target 품질 (PPL / KL / top-1 %): scalar RTN4 8.452 / 0.0027 / 97.7, RTN3 8.489 / 0.0100 / 95.8, 2+2 8.482 / 0.0111 / 95.6,
  D32 t16 8.487 / 0.0065 / 97.3, D32 t12 8.535 / 0.0119 / 96.3.
- 해석:
  - **같은 bit의 target 품질은 일반 scalar RTN이 RQ보다 좋음** (4bit KL 0.0027 vs 0.0065; 3bit 0.0100 vs 0.0119).
    앞서 "4bit에서 RQ target이 scalar보다 낫다"는 간격이 넓은 중첩 2+2와 비교한 결과였음.
  - RQ의 이득은 중첩: draft를 따로 저장하지 않고도 수락률이 높음. 저장 4bit의 RQ(2bit draft 95.2%)가 저장 6bit의 scalar(92.0%)를 이김.
    다만 target 품질은 scalar 4bit가 더 좋으므로 "같은 저장량·같은 target 품질"의 Pareto 비교가 필요.
  - RQ 1.5bit draft ≈ scalar 2bit draft.

### 저BPA target ≤ 2bit (2026-10-23, `results/specdec/t2/`, `specdec/sweep_t2.sh`)
- target 품질 (PPL / KL / top-1 %), scalar는 group 128 RTN(+0.25bit scale/zero, RQ는 코드북 비용 미포함):

  | bit | scalar RTN | D32 RQ | D16 RQ |
  |---|---|---|---|
  | 1 | 붕괴 | 8.979 / 0.0704 / 91.2 | SV8 9.048 / 0.0797 / 90.5 |
  | 1.5 | – | 8.766 / 0.0426 / 93.1 | – |
  | 2 | 8.966 / 0.0875 / 88.3 | 8.655 / 0.0270 / 94.4 | 8.719 / 0.0341 / 93.7 (SV8 0.0363) |
  | 3 | 8.489 / 0.0100 / 95.8 | 8.535 / 0.0119 / 96.3 | 8.577 / 0.0164 / 95.7 |
  | 4 | 8.452 / 0.0027 / 97.7 | 8.487 / 0.0065 / 97.3 | 8.513 / 0.0095 / 96.7 |

  scalar와 RQ의 교차점은 2~3bit. 2bit에서 RQ KL은 scalar의 1/3, **RQ 1bit target이 scalar 2bit target보다 좋음**.
- 수락률 % γ=1/2/4: D32 target 2bit(t8) + draft 0.5 / 1 / 1.5bit = 90.5 / 86.6 / 77.6, 93.9 / 91.9 / 86.5, 97.2 / 95.7 / 93.3.
  target 1.5bit(t6) + draft 0.5 / 1bit = 92.8 / 89.6 / 82.4, 97.0 / 94.4 / 90.9. D16 target 2bit + draft 1bit = 94.4 / 91.8 / 87.5.
  0.5bit draft + 2bit target ≈ 중첩 scalar 2+2(저장 4bit)의 수락률.

### E1 코드워드 포맷 (2026-10-24, `results/specdec/cwfmt/`, `specdec/sweep_cwfmt.sh`, `specrq/formats.py`)
- 학습된 D32 코드북의 unit-norm shape를 사후에 포맷 변환합니다(codeword마다 absmax scale, gain은 fp16, 모든 stage).
  인코딩은 변환된 codeword로 합니다. 다만 코드북을 그 포맷에 맞춰 다시 학습한 것은 아닙니다(format-aware 학습은 E2).
  측정은 PG-19 8k 10개, KL(exact‖target)입니다.

  | target | fp16 | fp8 | int8 | nf4 | fp4 | int4 | int3 |
  |---|---|---|---|---|---|---|---|
  | t4 (1 bit) | .0704 | .0707 | .0703 | .0713 (+1%) | .0726 (+3%) | .0727 (+3%) | .0814 (+16%) |
  | t8 (2 bit) | .0270 | .0270 | .0271 | .0285 (+6%) | .0299 (+11%) | .0304 (+13%) | .0386 (+43%) |
  | t16 (4 bit) | .0065 | .0066 | .0066 | .0081 (+25%) | .0086 (+32%) | .0089 (+37%) | .0132 (+103%) |

- 8bit 포맷은 손실이 없습니다. 4bit 포맷의 손해는 target 정밀도가 높을수록 커집니다(1bit +1~3%, 4bit +25~37%). 4bit 포맷 중에서는 nf4가 가장 좋습니다.
- 원인(KV 재구성 상대오차, 2K 프롬프트 1개): 순차 RQ 인코딩에서는 앞 stage의 포맷 오차가 잔차로 넘어가고, 뒤 stage가 그것을 흡수합니다.
  그래서 shape 오차가 9%(nf4)여도 KV 오차는 1~5%만 늘어납니다(L15 key t16: .060 → .061).
  남는 것은 바닥(floor) 오차뿐입니다. 이 바닥은 RQ 오차가 작은 곳에서 드러납니다. 높은 t가 그런 경우이고, L0 key도 그렇습니다(t16 .007 → .055).
  따라서 이 효과는 데이터셋(PG-19)이 쉬워서 생긴 것이 아닙니다.
- H3 미리보기 (t16에서 앞 s stage만 저정밀, 나머지는 fp16): nf4 x4 .0078, nf4 x8 .0079 (전체 nf4 .0081), int3 x4 .0136, int3 x8 .0129 (전체 int3 .0132).
  **뒤 stage를 fp16으로 되돌려도 개선이 없습니다.** 따라서 손해는 앞 stage의 포맷 오차에서 나옵니다.
  이 오차는 뒤 stage가 흡수하지 못합니다. 뒤 stage 코드북은 fp16 prefix의 잔차 분포에 맞춰 학습되어 있어서, prefix 포맷 오차가 만드는 잔차를 표현하지 못하는 것으로 보입니다.
  위에서 말한 "흡수 + 바닥" 해석은 이렇게 고쳐야 합니다. 바닥 오차는 마지막 stage에서 오는 것이 아니라, 앞 stage에서 흡수되지 않고 남은 오차입니다.
  → 사후 변환만으로는 "prefix 저정밀"이 공짜가 아닙니다. 비대칭 구성이 이득을 내려면 **format-aware 학습(E2)**이 필요합니다. 포맷을 적용한 prefix로 뒤 stage를 학습하는 것입니다.
- exact window를 끈 경우(`--rq-window 0`, t8): KL fp16 .2603 / nf4 .2783 (+7%) / int3 .6309 (+142%), top-1 77.1 / 76.1 / 64.1.
  기본 설정에서 최근 128 토큰 exact window는 RQ 오차 자체를 10배 가리고 있었습니다(.027 → .260).
  nf4의 상대 손해는 window가 있을 때와 비슷합니다(+6% → +7%). 반면 int3는 window가 없으면 훨씬 커집니다(+43% → +142%).
  아직 확인하지 않은 것: 장거리 검색 task.

### 다음 단계
1. 모델을 정합니다. 계획은 MHA인 Llama-2-7B를 주력으로, GQA 비교군은 Llama-3-8B로 하는 것입니다. 이어서 codebook 저장 경로를 정합니다.
2. 기준 codebook을 학습합니다(\(d=16\)과 \(d=8\), \(K=256\)).
3. `specrq/`에 format·codec·hook을 다시 구현합니다(key layout 보정 포함). 이어서 E1을 재실행하고 m=3을 추가합니다.
4. E0: 작은 draft–verify rollout harness를 만듭니다.
