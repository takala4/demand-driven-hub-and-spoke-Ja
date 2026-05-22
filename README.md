# 需要主導型の輸送ネットワーク形成： Hub and Spoke構造の創発

## 概要

本リポジトリでは，下記論文における数値計算コードを掲載している．

* 酒井 高良, 高山 雄貴: 需要主導型の輸送ネットワーク形成：Hub and Spoke構造の創発, 土木学会論文集, Vol. 81, No. 6, 24-00167, [https://www.jstage.jst.go.jp/article/jscejj/81/6/81_24-00167/_article/-char/ja/](https://www.jstage.jst.go.jp/article/jscejj/81/6/81_24-00167/_article/-char/ja/)
* 酒井 高良, 高山 雄貴: 需要主導型の輸送ネットワーク形成：Hub and Spoke構造の創発 (preprint), 2024, [https://doi.org/10.51094/jxiv.749](https://doi.org/10.51094/jxiv.749)


## 四つの実装

本リポジトリには同一問題に対する 4 つの実装を含めている．論文掲載 6 ケース (N=17) では，**v1, v2 は厳密に opt_Z と SP_tree が一致**することを確認済み．v3 は確率的ヒューリスティック，v4 は決定的 ε-最適である（後述）．

| | ファイル | 種別 | 説明 |
|---|---|---|---|
| **v1** | `code/hubspoke_v1.py` | 厳密 (paper-faithful) | 論文の Soland 型分枝限定法をそのまま実装．アルゴリズム検証のベースライン |
| **v2** | `code/hubspoke.py` | **厳密 (improved)** | v1 と同じ BB を Numba JIT 化 + 探索フィルタ追加で大幅高速化．**論文 6 ケースを最も高速・確実に証明** |
| **v3** | `code/hubspoke_v3.py` | **確率的ヒューリスティック** | Variable Neighborhood Search．大域収束保証なし．大規模 N で v2 が破綻する領域で実用解を秒オーダで取得 |
| **v4** | `code/hubspoke_v4.py` | 決定的 ε-最適（Gurobi 必要） | 凹費用を piecewise linear で近似し MILP として Gurobi で解く．流量が ν の整数倍に限られるので breakpoints を {0, ν, 2ν, ...} に置けば PWL は exact |

### v2: 改良された分枝限定法（本命）

v1 と完全に同じアルゴリズム（Soland 型 spatial BB）を以下で高速化：

* **Numba JIT 化**：per-leaf solve（dense Dijkstra + 線形化 + 流量伝播 + 下界計算）を 1 パスに融合
* **skip-on-push**：子の下界 `app_Z ≥ opt_Z` の leaf は heap に push しない
* **periodic heap rebuild**：`opt_Z` 改善時に heap 内 dead entry を一括削除
* **diff-based heap encoding**：heap entry を親への差分参照で表現してメモリ大幅圧縮
* **exact-leaf skip**：線形緩和が exact (`Z ≈ app_Z`) な leaf は push しない（degenerate ループ回避）

最遅 case (d) で legacy v1 の 1 時間 7 分 → **94 秒（v2）**．論文 6 ケースを全て厳密証明可能．

### v3: VNS（確率的ヒューリスティック）

* 解 = ノード 0 を根とする有向木（各ノードの親配列）
* 1-opt 局所探索 + shake (k 個親同時変更) + random restart
* **大域収束保証**：`restarts > 0` のとき Brimberg-Mladenović (1996) の漸近収束．時間 → ∞ で確率 1 で大域最適到達．**有限時間保証なし**
* 論文 6 ケース全てを 10〜30 秒で論文値発見（**証明はしない**）．N=50 (L=2401) でも 30 秒で良質解
* 厳密証明が要らない探索フェーズや大規模実験に有用

### v4: Gurobi MILP + PWL（決定的 ε-最適）

* 流量は ν の整数倍に限られる性質を利用し，breakpoints を `{0, ν, 2ν, ..., (N-1)ν}` に置くことで **PWL 近似は exact**
* Gurobi が分枝限定 + cuts + presolve で解く
* 論文 6 ケースのうち (a)(b)(e) は **数秒で証明完了**．(c)(f) は数十秒〜数分で 0.4-0.6% 範囲．**(d) は LP 緩和が loose で短時間に証明不可**（5 分で gap 7%）
* warm-start サポート（v3 解を渡すと incumbent として再利用）

### 実行時間の比較 (N=17, 論文 6 ケース)

実測値（同一マシンでの 1 回計測．v3 は確率的なので参考値）．**太字 = 厳密証明完了．それ以外は注釈参照**．

| ケース | v1 (paper-faithful) | v2 (improved BB) | v3 (VNS) | v4 (Gurobi MILP) |
|:-:|---:|---:|---:|---:|
| (a) | **< 1 s** | **0.7 s** | < 1 s †‡ | **0.4 s** |
| (b) | **< 1 s** | **1.3 s** | 30 s †‡ | **1.1 s** |
| (c) | **4 分 48 秒** | **10.5 s** | 10 s †‡ | 60 s (0.4% gap)§ |
| (d) | **1 時間 8 分** | **118 s** | 10 s †‡ | 120 s (24% gap), 5 分 (7.6% gap)§ |
| (e) | **29.7 s** | **2.3 s** | < 10 s †‡ | **4.9 s** |
| (f) | **16 分 44 秒** | **30.3 s** | < 10 s †‡ | 60 s (0.6% gap)§ |

† v3 は確率的ヒューリスティックなので大域最適性の証明はしない．これらの時間は **「論文 opt_Z に一致する解を発見した時間」** で，「証明された時間」ではない．seed や config で変動．
‡ 厳密一致は確認済み（論文値と `opt_Z` および `SP_tree` がビット単位で一致）．
§ v4 の "gap" は Gurobi MIPGap．time-limit 到達で打ち切り．**解は paper opt_Z に一致しているが LP 緩和側の closing が間に合わず証明未達**．時間を伸ばせば最終的に証明可能．


### スケーラビリティ（v3 のみ実測，参考値）

| 問題規模 | v2 (厳密 BB) | v3 (VNS heuristic) | v4 (Gurobi MILP) |
|---|---|---|---|
| N=17 case (d) | **94 s で証明** | 10 s で paper 値発見 | 5 分で gap 7.6% |
| N=25 case (d)型 | 困難（推定数時間以上） | **20 s で解** | 未確認（恐らく timeout） |
| N=50 case (d)型 | 不可能（指数爆発） | **34 s で解** | 未確認（恐らく timeout） |
| N=100+ | 不可能 | 数分で解 (証明なし) | 未確認 |


### 使い分け（要約）

| 用途 | 推奨 |
|---|---|
| **論文 N=17 全 6 ケース，厳密証明** | **v2** が最確実（最遅 case d も 94 秒） |
| 簡単な構造のケース (a)(b)(e) の厳密証明 | v4 が数秒 |
| 大規模 N で良質解（証明不要） | **v3 (VNS)** |
| 大規模 N + 厳密証明試行 | v4（時間多め），または将来の v5 (path-based) を待つ |


## リポジトリ構成

```
code/
├── hubspoke.py            # v2: 改良版 BB（本命，Numba JIT + 探索フィルタ）
├── hubspoke_v1.py         # v1: 論文通りの BB（参照用，アルゴリズム検証のベースライン）
├── hubspoke_v3.py         # v3: VNS 確率的ヒューリスティック
├── hubspoke_v4.py         # v4: Gurobi MILP + PWL 決定的 ε-最適
├── reproduce_paper.ipynb  # 論文掲載 6 ケースを一括再現する notebook
├── run.ipynb              # 単一ケース実行例（既存）
├── benchmark.py           # v1/v2/v3/v4 の性能比較
├── profile_run.py         # cProfile によるプロファイル
├── verify_small.py        # 小規模ケースでの実装間照合（subprocess 隔離 + メモリ上限）
└── _verify_worker.py      # verify_small.py のワーカ
```


## 使用方法

### 依存ライブラリ

* 必須: `numpy`, `pandas`, `networkx`, `matplotlib`, `numba`
* v3 と verify ハーネス: `psutil`
* v4: `gurobipy`（Gurobi ライセンスが必要．アカデミック無償ライセンス可）


### 論文の数値実験を再現

```
cd code
jupyter notebook reproduce_paper.ipynb
```

全 6 ケース (a)〜(f) を v2 で順に解き，論文掲載値との照合とネットワーク構造の可視化を行う．


### 単一ケースの実行

```
cd code
jupyter notebook run.ipynb
```


### v1〜v4 の性能比較

```
cd code
python benchmark.py all              # 全 6 ケース，v1/v2 比較
python benchmark.py "(e)"            # case (e) のみ
python benchmark.py all --no-v1      # v1 をスキップ（高速）
```


### 実装の正しさ検証（開発用）

実装に変更を加えた後，小規模ケースで結果が一致するか確認できる：

```
cd code
python verify_small.py 5                          # N=5, v2 vs v1（default）
python verify_small.py 10 --phi 50 --rho 2.5      # case (e) パラメータ
python verify_small.py 17 --impl v3 --phi 10 --rho 5  # v3 単独
python verify_small.py 17 --impl both --baseline v1 --impl v3  # 任意実装ペア比較
```

将来 `hubspoke_v5.py` 等の派生実装を追加する場合は `--impl v5` または `--impl hubspoke_<variant>` で対象を指定できる．worker の `IMPL_MODULES` 辞書にエイリアスを追加すると短縮形が使える．


## 連絡先

東京科学大学　[酒井高良](https://takala4.github.io/cv/)（sakai.t.dcad@m.isct.ac.jp）
