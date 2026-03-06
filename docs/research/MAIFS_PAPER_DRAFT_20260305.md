  # DAAC: Disagreement-Aware Adaptive Consensus for Multi-Agent Image Forgery Detection

  ## Abstract

  생성형 AI의 확산으로 이미지 위변조는 전통적 편집 조작에서 Diffusion·GAN 기반 완전 합성까지 다양해졌고, 단일 탐지 원리에 의존하는 탐지기는 도메인 편향을 피하기 어렵다. 이질적 단서를 각자 담당하는 다중 전문 에이전트를 조합하는 접근이 자연스러운 해법이나, 에이전트를 늘리는 것만으로는 충분하지 않다. 기존 trust 기반 규칙형 합의(COBRA 계열)는 사전 고정된 글로벌 trust 점수와 수식 기반 가중 집계에 의존하므로, 에이전트 간 구조적 불일치 패턴을 데이터로부터 학습하지 못한다는 한계가 있다.

본 논문은 에이전트 간 불일치 패턴을 43차원 메타 특징으로 명시적으로 학습하는 **DAAC(Disagreement-Aware Adaptive Consensus)**를 제안한다. 4개 전문 에이전트(Frequency, Noise, FatFormer, Spatial) 기반 MAIFS 플랫폼에서 고정 샘플셋(1,500장), 10-seed 반복 평가한 결과, DAAC-GBM은 macro-F1 0.861로 COBRA(0.266) 대비 유의하게 우수했다(p=0.00195). 에이전트별 맹점—Frequency/Noise/Spatial의 AI-Gen F1=0.000, FatFormer의 Manipulated F1=0.000—이 DAAC에서는 상보적 불일치 신호로 전환되어 세 클래스 모두에서 균형 잡힌 성능을 달성한다. 6개 데이터 조합 60회 반복에서도 DAAC는 COBRA 대비 +0.493의 우위를 일관되게 유지했다(p=1.63e-11). DAAC의 메타 분류기는 43차원 특징 위의 경량 GBM으로, 에이전트 inference 이후 추가되는 연산 부담이 미미하다.

  ## 1. Introduction

  이미지 위변조 탐지는 전통적 조작 탐지(IMDL)와 생성형 모델 기반 이미지 탐지(AIGC)를 동시에 만족해야 한다. 실제 운영에서는 JPEG 압축 흔적, 센서 노이즈 불일치, 공간적 조작 흔적, 생성 모델 아티팩트가 혼재하므로, 단일 모델의 편향은 오탐/미탐으로 직결된다. 이에 이질적 탐지 원리를 가진 다수의 전문 에이전트를 조합하는 다중 에이전트 접근이 자연스러운 해법으로 제시된다.

  그러나 **다중 에이전트 합의(consensus)** 단계가 병목이 된다. 기존 trust 기반 규칙형 합의(COBRA 계열: RoT/DRWA/AVGA)는 사전 고정된 글로벌 trust 점수를 기반으로 수식 기반 가중 집계를 수행한다. DRWA는 샘플별 분산으로 가중치를 조정하고 AVGA는 softmax temperature를 적응적으로 변경하지만, 이는 모두 handcrafted 규칙이며 에이전트들이 특정 클래스에 대해 가지는 구조적 맹점—예컨대 AI 생성 이미지를 감지 못하는 주파수 분석 에이전트, 조작 탐지 능력이 없는 생성 탐지 에이전트—의 패턴을 데이터로부터 학습하지 못한다. trust 프로파일을 oracle 수준으로 조정해도, 또는 알고리즘 변형 전반을 탐색해도 이 구조적 한계는 개선되지 않음을 실험적으로 확인한다.

  본 논문은 이 문제를 해결하기 위해 **DAAC(Disagreement-Aware Adaptive Consensus)**를 제안한다. DAAC의 핵심 아이디어는 "누가 맞았는가"가 아니라 **"에이전트들이 어떻게 서로 다르게 틀리는가"**를 학습 신호로 활용하는 것이다. 에이전트 판정·confidence·pairwise 충돌로 구성된 43차원 메타 특징 위에 GBM 분류기를 학습하면, 개별 에이전트의 클래스별 맹점이 오히려 풍부한 구분 신호로 전환된다.

  실험은 4개 전문 에이전트(Frequency/CAT-Net, Noise/MVSS, FatFormer, Spatial/Mesorch)로 구성된 MAIFS 시스템을 플랫폼으로 수행한다. MAIFS는 실험 인프라이며, **본 논문의 제안은 DAAC 합의 계층**에 있다.

  본 연구의 핵심 질문은 다음과 같다.

  - Q1. 규칙형 trust 합의(COBRA) 대비 불일치 메타학습(DAAC)이 유의하게 우수한가?
  - Q2. 성능 향상이 특정 split 우연 효과가 아닌가?
  - Q3. DAAC 성능에 기여하는 특징군은 무엇인가?
  - Q4. 개별 에이전트 단독 판정 대비 DAAC 앙상블의 이점은 어디서 비롯되는가?

  ## 2. Related Background

  ### 2.1 Multi-agent forensic aggregation

  다중 판정기 결합은 다수결/가중 다수결이 기본이다. 그러나 단순 가중 평균은 에이전트 간 상충(conflict) 구조를 잃고, 저성능 클래스가 전체 성능을 붕괴시키기 쉽다. 기존 연구는 주로 도구 수준의 성능 향상(더 좋은 탐지기)에 집중했으며, 합의 단계에서 불일치 패턴 자체를 학습 신호로 활용한 사례는 드물다.

  ### 2.2 COBRA-style trust-weighted consensus

  COBRA 합의는 에이전트별 신뢰 점수(trust score)를 기반으로 최종 판정을 수행하며, RoT/DRWA/AVGA 세 알고리즘 변형을 포함한다.

  - RoT: trust 임계값 기반 trusted/untrusted cohort 분할
  - DRWA: 분산 기반 동적 신뢰 가중치 조정
  - AVGA: 분산 의존 softmax attention 가중 합산

  COBRA는 빠른 rule-based 합의에 적합하나, 세 알고리즘 모두 사전 고정된 글로벌 trust 점수를 출발점으로 삼으며 수식 기반 집계 함수는 학습되지 않는다. DRWA/AVGA가 샘플별 분산 기반 조정을 수행하더라도, 이는 handcrafted 규칙으로 클래스별·도메인별 불일치 구조를 데이터로부터 학습하지 못한다는 근본 한계를 공유한다. 본 논문에서 COBRA(drwa)를 주요 비교 baseline으로 사용하고, trust 추정 방식 및 알고리즘 변형 전반을 진단 실험 대상으로 삼는다.

  ### 2.3 Meta-learning on disagreement

  DAAC의 핵심 차별점은 에이전트 출력(verdict, confidence)뿐 아니라 **에이전트 쌍 간 불일치 구조**를 명시적으로 특징화한다는 데 있다. pairwise 불일치, confidence 차이, 충돌 강도를 결합한 메타 특징 위의 분류기 학습은 도메인 이동에서도 상대 구조 신호를 유지한다.

  ## 3. DAAC Method

  ### 3.1 Problem formulation

  $n$개의 에이전트 $\{a_1, \ldots, a_n\}$ 이 이미지 $x$에 대해 판정 $v_i \in \mathcal{C}$ 와 confidence $c_i \in [0,1]$ 을 독립 산출한다고 가정한다. 최종 클래스 $\mathcal{C} = \{$`authentic`, `manipulated`, `ai_generated`$\}$ 에 대한 합의 함수 $f: \{(v_i, c_i)\}_{i=1}^n \to \mathcal{C}$ 를 학습하는 것이 목표다.

  COBRA는 $f$를 사전 고정된 trust 점수 기반의 handcrafted 수식으로 정의하지만, DAAC는 $f$를 데이터로부터 학습하며, 그 입력 표현을 **불일치 구조를 포함한 43차원 메타 특징**으로 설계한다.

  ### 3.2 DAAC feature design (A5, 43-dim)

  DAAC full feature(A5)는 총 43차원이다.

  - **Per-agent (20)**: verdict one-hot $4 \times 4 = 16$ + raw confidence 4
  - **Pairwise disagreement (18)**: $\mathbf{1}[v_i \neq v_j]$ 6쌍 + $|c_i - c_j|$ 6쌍 + $(c_i + c_j) \cdot \mathbf{1}[v_i \neq v_j]$ 6쌍
  - **Aggregate (5)**: confidence variance, verdict entropy, max-min confidence gap, unique verdict count, majority ratio

  핵심 설계 원칙: 원신호(개별 confidence)와 관계신호(쌍별 충돌)를 동시에 제공하되, trust를 메타 특징 안에 내재화하지 않는다. trust는 에이전트 출력 수집 단계에서 별도로 처리된다(trust-neutral confidence contract: `response.confidence = tool_conf`, trust는 COBRA 합의 단계에서 1회만 적용).

  ### 3.3 Learning models

  DAAC는 동일 43차원 특징셋에 대해 LR, GBM, MLP를 학습한다. **본 논문의 제안 모델은 DAAC-GBM**이며, XGBoost 백엔드(GPU)로 학습한다.

  Ablation 실험을 위해 5가지 특징 서브셋을 정의한다.

  | ID | 포함 특징 | Dim |
  |---|---|---:|
  | A1 | confidence only | 4 |
  | A2 | verdict one-hot only | 16 |
  | A3 | pairwise disagreement + aggregate | 23 |
  | A4 | verdict + confidence (per-agent) | 20 |
  | **A5** | **full DAAC** | **43** |

  ### 3.4 Experimental platform (MAIFS)

  DAAC의 입력인 에이전트 출력은 MAIFS의 4개 전문 에이전트에서 수집한다.

  | Agent | 분석 원리 |
  |---|---|
  | Frequency (CAT-Net) | JPEG 압축 흔적 탐지 |
  | Noise (MVSS) | PRNU/SRM 센서 노이즈 불일치 |
  | FatFormer | CLIP ViT-L/14 + DWT 기반 AI 생성 탐지 |
  | Spatial (Mesorch) | ViT 기반 부분 조작 영역 탐지 |

  에이전트 출력은 실데이터 precollected JSONL로 저장되어 재현성을 보장한다.

  ## 4. Experimental Setup

  ### 4.1 Data and protocol

  #### Protocol-P (주 실험)

  - 입력: precollected JSONL (`patha_agent_outputs_20260304_080157.jsonl`)
  - 전체 샘플 수: 1,500 (authentic / manipulated / ai_generated 균형)
  - split: train/val/test = 0.6/0.2/0.2
  - seeds: 300~309 (10회 반복)
  - 비교군: Majority Vote, Weighted MV, **COBRA(drwa)** (주 baseline), Logistic Stacking(A4+LR), DAAC(LR/GBM/MLP), 개별 에이전트 단독
  - 통계: paired Wilcoxon (vs COBRA)

  #### Protocol-M (다중 데이터 일반화)

  - 총 6개 데이터 조합 × 10 split-seed = **60 runs**
  - 3케이스 비교: COBRA-only vs DAAC-only vs COBRA+DAAC(fixed blend)
  - 통계: paired Wilcoxon, sign test (전체 60 runs 대상)

  **6개 데이터 조합 상세:**

  | 조합 | Authentic 출처 | Manipulated 출처 | AI-Generated 출처 | 검증 목적 |
  |------|--------------|----------------|-----------------|---------|
  | **DS-A** | CASIA2 Au | CASIA2 Tp (스플라이싱) | BigGAN | Protocol-P 기준 동일 출처 (scale 축소 검증) |
  | **DS-B** | ImageNet (자연 사진) | CASIA2 Tp (스플라이싱) | BigGAN | Authentic 출처 변경 시 일반화 |
  | **DS-C** | CASIA2 Au | IMD2020 (인페인팅) | BigGAN | Manipulated 조작 유형 변경 (스플라이싱→인페인팅) |
  | **DS-D** | ImageNet (자연 사진) | IMD2020 (인페인팅) | BigGAN | Authentic + Manipulated 둘 다 미지 도메인 |
  | **OpenSDI** | 소셜미디어 실세계 이미지 | OpenSDI 조작 이미지 | — | 실세계 분포, 학술 벤치마크 외 도메인 |
  | **AI-GenBench** | — | — | 다양한 생성 방식 proxy | BigGAN 외 AI 생성 방식으로 교체 |

  **각 조합의 검증 가설:**
  - DS-A~D: authentic/manipulated 출처를 순차적으로 교체하여 **데이터셋 cross-domain 일반화** 검증
  - DS-B vs DS-A: authentic 출처(CASIA2→ImageNet)가 바뀌어도 불일치 패턴이 유효한가
  - DS-C vs DS-A: manipulated 조작 유형(스플라이싱→인페인팅)이 바뀌어도 FatFormer 맹점 신호가 유지되는가
  - DS-D: authentic과 manipulated 모두 훈련 시 미지 도메인일 때 **최악 조건** 검증
  - OpenSDI: 학술 벤치마크가 아닌 **실세계 소셜미디어 분포**에서의 적용 가능성
  - AI-GenBench: BigGAN 이외 **다른 생성 방식**에서도 ai_generated 탐지가 가능한가

  ### 4.2 Anti-luck controls

  1. precollected 샘플셋 고정 후 split만 변경 (데이터 우연 효과 차단)
  2. 동일 split에서 모든 방법을 paired 비교
  3. 10-seed 반복 및 평균±표준편차 보고
  4. 다중 데이터 조합 확장(총 60 runs)

  ## 5. Results

  ### 5.1 Main comparison (Protocol-P, 10 seeds)

  | Method | Macro-F1 (mean±std) | p vs COBRA (Wilcoxon) |
  |---|---:|---:|
  | Majority Vote | 0.2774±0.0178 | 0.0371 |
  | Weighted Majority Vote | 0.2664±0.0128 | 1.0000 |
  | **COBRA (drwa)** | **0.2664±0.0090** | — (기준) |
  | Logistic Stacking (A4+LR) | 0.8500±0.0161 | 0.00195** |
  | DAAC-LR (A5+LR) | 0.8541±0.0087 | 0.00195** |
  | **DAAC-GBM (A5+GBM)** | **0.8613±0.0156** | **0.00195**** |
  | DAAC-MLP (A5+MLP) | 0.8570±0.0122 | 0.00195** |

  ** p < 0.01 (10 seeds, paired Wilcoxon)

  해석: DAAC 계열은 모든 10 seed에서 COBRA 대비 일관되게 우수하다. 가장 단순한 비교인 Logistic Stacking(A4+LR)조차 COBRA 대비 +0.584의 절대 우위를 보이며, 이는 불일치 특징 학습 자체의 기여임을 시사한다. 제안 모델 DAAC-GBM이 최고 성능을 달성한다.

  ### 5.2 Individual agent standalone performance

  개별 에이전트의 단독 판정 능력을 동일 10-seed 프로토콜로 평가하여 DAAC-GBM과 비교한다.

  | Method | Macro-F1 (mean±std) | p vs DAAC-GBM (Wilcoxon) |
  |---|---:|---:|
  | Frequency (단독) | 0.4284±0.0107 | 0.00195** |
  | Noise (단독) | 0.3445±0.0150 | 0.00195** |
  | FatFormer (단독) | 0.3385±0.0158 | 0.00195** |
| Spatial (단독) | 0.3083±0.0114 | 0.00195** |
| COBRA (drwa, 앙상블) | 0.2664±0.0086 | 0.00195** |
| **DAAC-GBM (앙상블)** | **0.8613±0.0148** | — |

** p < 0.01 (10 seeds, paired Wilcoxon)

해석: 개별 에이전트 단독 성능(0.31~0.43)은 DAAC-GBM(0.86) 대비 모두 유의하게 낮다(p=0.00195). 더 주목할 점은 COBRA 앙상블(0.2664)이 개별 에이전트보다도 낮다는 것이다. 단순 신뢰 가중 합산은 에이전트 간 맹점 갈등을 해소하지 못하고 오히려 성능을 떨어뜨릴 수 있음을 보여준다.

### 5.3 Per-class behavior

| Method | Authentic F1 | Manipulated F1 | AI-generated F1 |
|---|---:|---:|---:|
| Frequency (단독) | 0.804 | 0.481 | **0.000** |
| Noise (단독) | 0.558 | 0.476 | **0.000** |
| FatFormer (단독) | 0.542 | **0.000** | 0.473 |
| Spatial (단독) | 0.570 | 0.355 | **0.000** |
| COBRA (drwa) | 0.573 | 0.199 | 0.027 |
| Logistic Stacking | 0.809 | 0.787 | 0.954 |
| **DAAC-GBM** | **0.821** | **0.800** | **0.963** |

### 5.3.1 Precision / Recall 분리 (10-seed pooled)

단일 F1 지표의 한계를 보완하기 위해 DAAC-GBM과 COBRA의 클래스별 Precision / Recall을 분리하여 보고한다.

| Method | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| **DAAC-GBM** | authentic | 0.815 | 0.839 | 0.827 |
| | manipulated | 0.808 | 0.807 | 0.807 |
| | ai_generated | **0.972** | **0.945** | **0.958** |
| | **macro avg** | **0.865** | **0.864** | **0.864** |
| COBRA (drwa) | authentic | 0.399 | **0.998** | 0.570 |
| | manipulated | 0.247 | 0.119 | 0.161 |
| | ai_generated | 1.000 | 0.019 | 0.037 |
| | **macro avg** | 0.549 | 0.379 | 0.256 |

해석: COBRA는 authentic Recall=0.998로 거의 모든 이미지를 authentic으로 판정하는 편향을 보인다. ai_generated Recall=0.019는 1,000개 중 19개만 AI 생성으로 탐지함을 의미한다. 반면 DAAC-GBM은 세 클래스 모두 Precision과 Recall이 균형적으로 높으며, 특히 ai_generated에서 P=0.972 / R=0.945로 탐지 누락(false negative)이 매우 낮다.

### 5.3.2 Confusion Matrix (10-seed pooled)

**DAAC-GBM:**

|  | pred: authentic | pred: manipulated | pred: ai_generated |
|---|---:|---:|---:|
| **true: authentic** | **839** | 157 | 4 |
| **true: manipulated** | 170 | **807** | 23 |
| **true: ai_generated** | 20 | 35 | **945** |

**COBRA (drwa):**

|  | pred: authentic | pred: manipulated | pred: ai_generated |
|---|---:|---:|---:|
| **true: authentic** | **998** | 2 | 0 |
| **true: manipulated** | 881 | **119** | 0 |
| **true: ai_generated** | 621 | 360 | **19** |

해석: COBRA의 confusion matrix는 세 클래스가 모두 authentic으로 흡수되는 구조적 편향을 직접적으로 보여준다. 1,000개의 manipulated 샘플 중 881개(88.1%), 1,000개의 ai_generated 샘플 중 621개(62.1%)가 authentic으로 오분류된다. DAAC-GBM은 authentic↔manipulated 간 혼동(157/170)이 주요 오류 패턴이며, ai_generated 오분류는 55건으로 매우 낮다.

### 5.3.3 Cohen's Kappa

| Method | Cohen's Kappa | 해석 |
|---|---:|---|
| COBRA (drwa) | 0.0680 | 우연 수준에 가까움 (slight agreement) |
| **DAAC-GBM** | **0.7955** | 강한 일치 (substantial to almost perfect) |

해석: Cohen's Kappa는 클래스 불균형과 우연 일치를 보정한 지표다. COBRA의 κ=0.068은 random 분류기와 거의 차이가 없음을 의미하며, DAAC-GBM의 κ=0.796은 실질적으로 신뢰할 수 있는 분류기임을 나타낸다 (κ>0.8이 "almost perfect" 기준).

해석: 개별 에이전트는 특정 클래스에 대한 **구조적 맹점(blind spot)**을 가진다. Frequency/Noise/Spatial은 `ai_generated` F1=0.000으로 AI 생성 이미지를 탐지하지 못하고, FatFormer는 `manipulated` F1=0.000으로 조작 탐지 능력이 없다. COBRA는 이 맹점들을 가중 합산하므로 최종 판정에도 전파된다. DAAC-GBM은 에이전트 간 충돌 패턴을 학습하여 세 클래스 모두에서 균형 잡힌 성능을 달성한다.

### 5.4 Feature ablation (GBM)

| Ablation | Dim | Macro-F1 (mean±std) |
|---|---:|---:|
| A1 confidence only | 4 | 0.7972±0.0149 |
| A2 verdict only | 16 | 0.7596±0.0213 |
| A3 disagreement only | 23 | 0.8534±0.0102 |
| A4 verdict+confidence | 20 | 0.8506±0.0202 |
| **A5 full DAAC** | **43** | **0.8613±0.0156** |

해석: A3(불일치 + 집계, 23-dim)이 이미 0.8534로 높아, **불일치 구조 자체가 핵심 탐지 신호**임을 확인한다. A5가 A3 대비 추가 개선(+0.0079)을 제공하여 개별 confidence와 집계 통계가 불일치 신호를 보완적으로 강화함을 보인다.

### 5.5 Multi-dataset generalization (Protocol-M, 60 runs)

**데이터 조합별 결과:**

| 조합 | 검증 포인트 | COBRA Macro-F1 | DAAC Macro-F1 | Delta |
|------|-----------|---------------|--------------|-------|
| DS-A (CASIA+BigGAN) | 기준 조합 | 0.2771±0.0171 | 0.8353±0.0282 | +0.558 |
| DS-B (ImageNet+CASIA+BigGAN) | Authentic 도메인 변경 | 0.2738±0.0177 | 0.8672±0.0203 | +0.593 |
| DS-C (CASIA+IMD2020+BigGAN) | Manipulated 유형 변경 | 0.3027±0.0198 | 0.8976±0.0232 | +0.595 |
| DS-D (ImageNet+IMD2020+BigGAN) | 양쪽 모두 미지 도메인 | 0.2995±0.0224 | 0.8634±0.0151 | +0.564 |
| OpenSDI | 실세계 소셜미디어 분포 | 0.1772±0.0121 | 0.4094±0.0362 | +0.232 |
| AI-GenBench proxy | BigGAN 외 생성 방식 | 0.3332±0.0225 | 0.7502±0.0199 | +0.417 |

**전체 집계 (60 runs):**

| Case | Macro-F1 (mean±std) |
|---|---:|
| COBRA-only | 0.2772±0.0525 |
| DAAC-only | 0.7705±0.1710 |
| COBRA+DAAC (fixed blend) | 0.6818±0.2229 |

paired 비교(전체 60 runs):

- DAAC vs COBRA: +0.4933, Wilcoxon p=1.63e-11, sign pos/neg=60/0
- COBRA+DAAC vs COBRA: +0.4045, Wilcoxon p=1.99e-11, sign pos/neg=58/2
- COBRA+DAAC vs DAAC: −0.0888, Wilcoxon p=1.63e-11, sign pos/neg=0/60

해석:

- **DS-A~D (학술 벤치마크 계열)**: authentic 출처(CASIA2→ImageNet), manipulated 유형(스플라이싱→인페인팅)이 바뀌어도 DAAC는 0.835~0.898의 높은 성능을 유지한다. 이는 FatFormer↔나머지 에이전트 간의 불일치 신호가 데이터 출처와 무관하게 일관되게 발생하기 때문이다. DS-D(양쪽 모두 미지 도메인)에서도 delta +0.564로 우위가 유지되어, 학습 시 미지 도메인에서도 불일치 패턴 학습이 유효함을 확인한다.
- **OpenSDI (실세계 분포)**: 학술 벤치마크 대비 성능이 낮아지나(DAAC 0.409), COBRA(0.177) 대비 +0.232의 우위는 유지된다. 실세계 분포에서 절대 성능 하락은 에이전트 자체의 out-of-distribution 문제이며 합의 방식의 한계는 아니다.
- **AI-GenBench (다른 생성 방식)**: BigGAN이 아닌 다른 AI 생성 방식에서도 DAAC(0.750)가 COBRA(0.333) 대비 +0.417 우위를 보여, BigGAN 특화가 아닌 일반적 ai_generated 불일치 패턴을 학습했음을 시사한다.
- **고정 블렌딩의 한계**: COBRA+DAAC(fixed 0.5)는 모든 60 runs에서 DAAC 단독보다 낮다(sign 0/60). 도메인별 최적 blending 비율이 다르므로 고정 혼합은 일관된 열위를 보인다.

### 5.6 COBRA structural diagnosis

COBRA 저성능이 단순 설정 불일치 때문인지 분리한다. 동일 JSONL/동일 10 split-seed에서 trust 프로파일·알고리즘·파라미터를 전수 교차 비교한다.

기준: DRWA + static trust = 0.2664 ± 0.0090

| Setting | Macro-F1 (mean±std) | Delta | Wilcoxon p |
|---|---:|---:|---:|
| DRWA + static trust (baseline) | 0.2664±0.0090 | +0.000 | 1.0000 |
| DRWA + metrics-derived trust | 0.2656±0.0106 | −0.001 | 0.6523 |
| DRWA + oracle (test upper) trust | 0.2649±0.0098 | −0.002 | 0.4316 |
| DRWA + shuffled trust | 0.2660±0.0091 | −0.000 | 0.2383 |
| AVGA + static trust | 0.2780±0.0177 | +0.012 | 0.0371 |
| AUTO + static trust | 0.2763±0.0171 | +0.010 | 0.0840 |

민감도: DRWA epsilon, RoT threshold, AVGA temperature 전 범위 탐색 시 최대 delta +0.0116.

해석: oracle trust를 포함한 어떤 trust 추정 방식도 COBRA 성능을 의미 있게 회복하지 못했다. 설정 튜닝 상한(+0.0116)은 DAAC-GBM(0.8613)과의 격차(−0.595)를 전혀 설명하지 못한다. **COBRA의 구조적 한계는 trust 품질의 문제가 아닌 합의 방식 자체의 표현력 부족**에 기인한다.

## 6. Discussion

### 6.1 Why COBRA underperforms: structural limitations

COBRA가 낮은 이유는 다음 요인이 결합된 결과다.

1. 글로벌 신뢰 점수는 클래스별 오류 구조를 반영하지 못함
2. 단일 합의 규칙은 에이전트 간 상충 구조를 표현하지 못함
3. AI-generated 클래스에서 3개 에이전트가 공통으로 맹점을 가지므로, 가중 합산이 오판을 증폭

특히 Section 5.6 진단에서 oracle trust 조건에서도 F1이 오히려 소폭 하락(−0.0015)하는데, 이는 단순 가중 합산의 표현력 한계가 trust 품질보다 지배적임을 의미한다.

### 6.2 Agent blind spots and DAAC's complementary recovery

4개 에이전트는 각자 뚜렷한 맹점을 가진다.

- **Frequency, Noise, Spatial**: `ai_generated` F1=0.000
- **FatFormer**: `manipulated` F1=0.000

이 맹점들은 **상보적**이다. Frequency/Noise/Spatial이 manipulated를 인식할 때, FatFormer는 AI-generated를 인식한다. DAAC의 핵심 특징 `disagree_frequency_fatformer`(pairwise 불일치)는 이 정반대 판정 패턴을 직접 포착하며, feature importance 분석에서 56.5%를 차지하는 최상위 특징이다. DAAC는 에이전트 성능을 직접 높이는 것이 아니라, **에이전트들이 서로 다르게 틀리는 구조**를 학습하여 맹점을 보완한다.

### 6.3 Why disagreement features are sufficient

A3(불일치만, 23-dim) F1=0.8534는 A4(verdict+confidence, 20-dim) F1=0.8506보다 높다. 개별 판정·confidence 정보보다 **불일치 패턴 자체**가 더 강한 클래스 구분 신호임을 의미한다. 이는 에이전트 맹점이 뒤집어 보면 클래스 탐지 신호라는 DAAC의 핵심 통찰을 뒷받침한다.

### 6.4 Fusion implication

COBRA+DAAC 고정 혼합 가중치는 모든 60 runs에서 DAAC 단독보다 낮았다(sign 0/60). 향후 샘플별 uncertainty-aware gating 또는 학습된 adaptive fusion이 필요하다.

## 7. Limitations

1. 비교는 MAIFS 내부 프로토콜 중심이며, 외부 SOTA 탐지기와의 완전 동일 조건 재평가는 아직 제한적이다.
2. OpenSDI/AI-GenBench는 subset/proxy 구성으로 사용되었다.
3. COBRA는 `drwa` 알고리즘을 중심 baseline으로 사용했으며, 알고리즘·신뢰 프로파일 전수 탐색 결과는 Section 5.6 별도 진단으로 분리된다.
4. 현재 에이전트(FatFormer/Noise/Frequency/Spatial)는 BigGAN 기반 데이터셋 중심으로 학습되어 있어, OpenSDI(SD/Flux 계열) 등 최신 Diffusion 이미지에 대한 에이전트 단독 성능이 제한적이다. DAAC 합의 계층의 우위는 유지되나 절대 성능 향상을 위해서는 에이전트 재학습이 필요하다.

**후속 연구 과제:**
- FatFormer를 OpenSDI(SD1.5/SD2.1/SDXL/SD3/Flux.1) 데이터로 fine-tuning 후 DAAC 메타 분류기 재학습, 성능 변화 측정
- 에이전트 재학습 시 불일치 패턴 구조의 변화 분석 (disagree_frequency_fatformer 중요도가 유지되는지 여부)

## 8. Conclusion

본 논문은 다중 에이전트 이미지 포렌식에서 에이전트 간 불일치 패턴을 명시적으로 학습하는 합의 메커니즘인 DAAC를 제안했다. 43차원 메타 특징(per-agent, pairwise disagreement, aggregate)을 GBM 분류기로 학습한 DAAC-GBM은 정적 신뢰 가중 합의(COBRA) 대비 macro-F1 기준 대폭 우수하며(0.861 vs 0.266, p=0.00195), 6개 데이터 조합 60회 반복에서도 일관된 우위를 확인했다(sign 60/0). 핵심 발견은 개별 에이전트가 가지는 클래스별 맹점이 서로 상보적이며, DAAC는 이 상보적 불일치 구조를 탐지 신호로 전환한다는 것이다. 에이전트 성능 개선 없이 합의 계층만 개선해도 +0.43~+0.55의 F1 향상이 가능함을 보였으며, 이는 다중 에이전트 시스템에서 합의 방식의 설계가 개별 에이전트 성능만큼 중요함을 시사한다.

## 9. Reproducibility Appendix

### 9.1 Main comparison tables (Protocol-P)

```bash
python experiments/run_paper_final.py experiments/configs/paper_final.yaml
```

### 9.2 Individual agent evaluation

```bash
python experiments/run_agent_eval.py experiments/configs/paper_final.yaml
```

### 9.3 Multi-dataset generalization (Protocol-M)

```bash
python experiments/summarize_case3_multidata.py \
  --output-json experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260304_v2_6sets.json \
  --output-md experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260304_v2_6sets.md
```

### 9.4 COBRA structural diagnosis

```bash
python experiments/run_cobra_mismatch_diagnostics.py experiments/configs/paper_final.yaml
```

### 9.5 Primary result artifacts

- `experiments/results/paper_final/paper_final_paper_final_20260304_145652.json`
- `experiments/results/agent_eval/agent_eval_paper_final_20260305_053951.json`
- `experiments/results/phase2_patha_case3_multidata/multi_dataset_case3_comparison_20260304_v2_6sets.json`
- `experiments/results/cobra_diagnostics/cobra_mismatch_diagnostics_20260305_052144.json`
