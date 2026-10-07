# Handoff v17 — IPC 全層事故：判層、找舊案、套 SOP、掉人工再長回來（提案，2026-10-07，未跑）

給接手的 agent：你接的是一個**提案**，不是跑過的實驗。前十六輪（`README.md` §2）把兩顆模型的脾氣摸清楚了，這一份把它們放到一個真實系統裡：
一群跑 Ubuntu + microk8s + 業務 app 的 IPC，硬體到業務全部進 LGTM，出事時系統要回答四件事——

1. **哪一層出問題**（硬體／OS／microk8s／app／業務／證據不足）。
2. **跟過往有沒有關**：同一件事、同徵狀不同原因、還是沒見過。
3. **能不能用同一份 SOP 修這台**：直接套、改參數套、不適用。
4. **都不像的掉人工**：人找原因、新增分類或 SOP，系統把這一案存回去，下次同樣的事發生在另一台就找得到。

這是老方法「案例推理」（retrieve → reuse → revise → retain），檢索用 EmbeddingGemma 2、判斷用 Gemma 4 26B-A4B 讀字母、E4B 當第一關。**不訓練任何模型**；系統靠案例庫長大。

第一天做什麼：讀本檔 §1–§3 和 §10 的 Phase 0；Phase 0 沒有模型，先定分層、定卡、手填 30 張歷史案例卡。模型是 Phase 2 的事。

## 0. 規則（承 `handoff-gb10.md` §0）

1. 真實事故資料（卡、日誌、主機名）**不離開 GB10**，不上雲、不進 git。repo 裡只放 schema、程式、彙總數字。
2. 每個 Phase 先寫門檻再做（§10 已寫死），做完照門檻填 §14；沒過照實寫。
3. 不訓練、不改 `decide/prompt.py` 的模板與 `SYSTEM`。門檻用校準後信心 + 錯誤預算（`bench/thresholds.py`），寫進 lock 檔，`verify.py --check-lock` 當 CI。
4. 模型只讀「案例卡」（§5），不讀原始日誌。卡由程式照固定查詢生成，人可改。
5. 自動執行只限**可逆**的 SOP；不可逆的只建議。
6. 150 筆以下的樣本只報個數不報百分比（v15／v16 的教訓：150 則的解析度是 ±3 則，從 3 則的差距讀故事會錯）。

## 1. 一句話設計

```
OTel/Alloy → Loki · Mimir · Tempo → Grafana Alerting → Keep（去重、規則關聯、開事故）
                                                          │ workflow: HTTP 呼叫
                                                          ▼
                              判層服務（GB10）：卡生成器 → EG2 找舊案／novelty → E4B 先答 → 26B 判四題
                                                          │ enrichment 欄位、建議、信心
                                                          ▼
                     Keep 事故頁（人看、人按）→ 可逆 SOP 自動跑 / 其餘人工 → 人填根因與 SOP → 存回案例庫與 SOP 目錄
```

模型的位置是**事故層級**，不是 alert 層級：一個事故一張卡、幾次讀取，延遲預算是秒，不是毫秒。alert 層級的去重、分組交給 Keep 與 Alertmanager 的規則（`results/10-landscape.md` §10 討論過為什麼那裡不需要模型）。

## 2. 從十六輪借來的發現，在這裡怎麼用

| 發現（輪） | 在這裡的用法 |
|---|---|
| 答案寫在文字裡的判斷題，26B 零樣本七類 ≥ 0.98（v1–v3）；答案不在文字裡的預測題接近擲硬幣（optjev，AUROC 0.475） | 四道題全部設計成「讀卡就能答」。卡上沒有的事（真正的新根因）不問模型，交人 |
| 判準寫成可數規則，弱題 0.81 → 0.885、0.815 → 0.95（v2 補） | 每層的判準寫成可核對的條件（「溫度或 SMART 或 ECC 先異常，且早於其他層」），不用形容詞 |
| raw 信心永遠 0.99，門檻要校準；錯誤預算反推門檻 + lock 檔（v1、v7、v9） | 每道題各自一個溫度與門檻，ε 自訂（判層 5%、自動套 SOP 2%）；沒把握就交人 |
| E4B 先答、34% 送 26B，準確率不變，26B 負載剩三分之一（v3） | E4B 當第一關，門檻 0.999；GB10 上兩顆都在 |
| 選項順序敏感：弱題兩成會翻（v5） | 四道題的選項順序寫死，校準用同一順序 |
| 選項 > 10 用淘汰賽，照目錄順序分組，11 次讀取 0.947；EG 前 K 名當前置追不上（v14–v16） | SOP 目錄超過 10 份時用 `bench/route_bench.py` 的淘汰賽；**先用結構欄位（層、設備型號）過濾**，通常剩不到 10 份，一次讀完 |
| EG2 比「像不像」不比「對不對」；例子比例子才準，口語比條文不準；recall@20 0.953（v13–v14） | EG2 只做卡對卡的檢索與 novelty，不下判斷；索引裡是歷史卡（例子），不是 SOP 條文 |
| 打散分組沒用，錯誤跟著題目走（v16） | 不在檢索排序上花力氣；力氣花在卡的品質 |
| 思考只對要數點數的題有用，對急迫度有害，每題 11–19 秒（v12 E4） | 自動流程不開思考；只在「掉人工」時開思考列可能原因給人參考，標明是建議 |
| 31B 準 2 點但慢 3 倍、12B 不省時間（v12 E3） | 維持 26B-A4B；不換模型 |
| `<bos>`、`--swa-full`、slot context、`--reasoning-budget 0`（v6、v11、v12） | 判層服務用 llama-server ≥ b11371，旗標照 `handoff-gb10.md` §7 |
| packed readout 後面的題會翻、JSON prefill 沒用（v5、v7） | 四道題分開讀，共用 state（卡）吃前綴快取 |
| Clef 判斷頭接不上引擎、jevify LoRA 比零訓練差（v10、v11） | 不訓練、不接第三方判斷頭 |
| Gemini 盲寫資料避免同源；但合成 storm 會自證（v3、v14；`10-landscape.md` §10） | 案例卡只能來自真實事故；合成只能用來測卡生成器的格式，不能用來算準確率 |
| 事後解釋會錯（v15 → v16） | 門檻先寫；事後診斷另起一節標明「不是門檻」 |

## 3. 分層與每層的證據

五層加一個「證據不足」。每層列子類與固定查詢；子類是第一題選項底下的說明，也是案例卡的 `layer_detail` 欄位值域。

| 層 | 子類 | 證據（LGTM 查詢的來源） |
|---|---|---|
| **H 硬體** | 過熱降頻、磁碟 SMART／壞軌、記憶體 ECC、電源／斷電重啟、網卡實體 | node_exporter（`node_hwmon_temp_celsius`、`node_edac_*`、`node_boot_time_seconds` 跳變）、smartctl_exporter、dmesg（Loki，Alloy 的 journald/`kmsg`） |
| **O OS（Ubuntu）** | 磁碟滿、OOM killer、時間不同步、網路設定／DNS、snap 自動更新、systemd 服務失敗 | node_exporter（`node_filesystem_avail_bytes`、`node_vmstat_oom_kill`、`node_timex_*`）、journald（`_SYSTEMD_UNIT`、`snapd`） |
| **K microk8s** | kubelet／containerd 異常、CrashLoopBackOff、PVC 掛不上、CoreDNS、calico、**snap refresh 重啟 microk8s**、單節點 dqlite 卡住、憑證過期 | kube-state-metrics（`kube_pod_container_status_restarts_total`、`kube_pod_status_phase`）、kubelet／containerd journald、`microk8s inspect` 輸出（事故時由 SOP 或程式收） |
| **A App** | 5xx／錯誤率、延遲、連不到依賴（DB、MQTT、PLC gateway）、設定錯、剛升版、資源不足被 throttle | app 日誌（Loki，Drain 模板）、Tempo（錯誤率、延遲、service graph 的依賴邊）、`container_cpu_cfs_throttled_*`、部署事件 |
| **B 業務** | 產出掉、訊息積壓、設備離線、資料不更新 | 業務指標（Mimir）、PLC／設備連線指標、業務日誌 |
| **U 證據不足／跨層** | 卡上沒有任何一層有明確異常，或多層同時且無法分先後 | — |

**根因方向的先驗**：異常通常由下往上傳（H → O → K → A → B）。卡上把每層「第一次異常的時間」排成時間軸；最早的那層是強線索，寫進第一題判準，但不是鐵律（app 把磁碟寫滿是 A 造成 O）。

## 4. 案例卡（模型唯一讀的東西）

一張卡 200–400 字，固定格式。由 `card_builder` 在事故開啟時對五層跑固定查詢產生，人可以在 Keep 事故頁補欄位。

```yaml
case_id: 2026-10-07T03:14Z-ipc-17          # 時間 + 設備
device: {model: "IPC-XXXX", site: "L3", ubuntu: "24.04.2", microk8s: "1.32/stable", app: "lineagent 2.8.1"}
window: {start: "03:02", end: "03:20"}      # 第一個 alert 前 10 分鐘到開卡
alerts:                                     # Keep 事故裡的 alert，去重後，依時間
  - {t: "03:12", src: mimir, name: KubePodCrashLooping, labels: {ns: line, pod: lineagent-*}}
  - {t: "03:14", src: loki,  name: AppErrorBurst, template: "dial tcp <IP>:5432: connect: connection refused"}
  - {t: "03:15", src: biz,   name: ThroughputDrop, value: "-62% vs 1h"}
layers:                                     # 每層固定查詢的結果；查不到就寫 none，不留空
  H: {first_anomaly: null, temp_max_c: 61, smart_errors: 0, ecc: 0, reboot: false, data: ok}
  O: {first_anomaly: "03:09", disk_avail_pct: 2, oom_kills: 0, ntp_offset_ms: 3, snap_refresh: null, data: ok}
  K: {first_anomaly: "03:11", restarts_15m: 7, pods_not_ready: ["postgres-0"], pvc_pending: 0, coredns_err: 0, microk8s_restart: false, data: ok}
  A: {first_anomaly: "03:12", err_rate_pct: 48, p95_ms: 2100, deps_failing: ["postgres:5432"], top_templates: ["dial tcp <IP>:5432: connection refused", "retry <N>/5"], deploy_recent: false, data: ok}
  B: {first_anomaly: "03:15", throughput_delta_pct: -62, devices_offline: 0, data: ok}
timeline: ["03:09 O disk 2%", "03:11 K postgres-0 NotReady", "03:12 A conn refused", "03:15 B throughput -62%"]
missing: []                                 # 哪些查詢沒有資料（Loki 沒收到、exporter 掛了）
# 以下由人填（Phase 0 手填；上線後在 Keep 事故頁填）
human: {layer: "O", layer_detail: "disk_full", root_cause: "日誌輪替壞掉，/var 滿，postgres 寫不進去", sop: "SOP-O-003", outcome: "fixed", notes: ""}
```

卡生成器的規則：
- 查詢清單寫死在 `cbr/queries.yaml`（PromQL／LogQL／TraceQL 各一組），版本化；改清單要重跑歷史卡。
- 日誌只放 **Drain 模板**（Loki 的 pattern 偵測或自己跑 Drain3），變數（IP、數字、pod hash）挖掉；原始行留在 Loki，卡上放連結。
- `missing` 一定要填；第一題的「證據不足」選項要靠它。
- 數值用相對變化（比 1 小時前）而非絕對值，不同台才能比。
- 卡上**不放** runbook 連結、dashboard 連結、長 description。

## 5. 四道選擇題（26B，E4B 先答）

全部用 `decide/prompt.py` 的 `TemplateRenderer.render(state, question)`，`state` 是卡的 YAML 原文，`question = {instructions, options: {A: {label, criteria}}}`；讀法 `decide/client.py::read_option_probs`。選項順序寫死。

**Q1 根因在哪一層**（ε = 5%）
```
instructions: 這張事故卡的根因最可能在哪一層？
A 硬體：溫度／SMART／ECC／電源有異常，且 first_anomaly 不晚於其他層
B OS：磁碟、OOM、時間、網路設定或 snap 更新有異常，且不晚於 K/A/B
C microk8s：kubelet／containerd／pod／PVC／CoreDNS／microk8s 重啟有異常，H 與 O 沒有更早的異常
D App：錯誤率或延遲異常、或依賴連不上，而 H/O/K 沒有更早的異常；deps_failing 指向的服務若在 K 層 NotReady，不選 D
E 業務：只有業務指標異常，其他層正常
F 證據不足或跨層：missing 非空且涵蓋可疑層，或兩層以上同時異常無法分先後
```

**Q2 跟舊案 X 是不是同一件事**（ε = 5%；X 由 EG2 給前三名，每個候選各問一次，state = 兩張卡並排）
```
instructions: 新卡與舊卡 X 是不是同一件事？
A 同一根因：同層同子類，關鍵證據（同一依賴、同一模板、同一資源耗盡）一致，版本差異不影響
B 同徵狀不同原因：alerts 與業務影響相似，但 layers 的第一異常層或關鍵證據不同
C 不同：層或徵狀都不同
```

**Q3 SOP Y 適不適用這台**（ε = 2%，因為會自動執行）
```
instructions: 依 SOP Y 的前提條件逐條對照這張卡，SOP Y 適不適用？
A 直接套：前提全部成立，版本在 SOP 標明的範圍內
B 改參數後套：前提成立，但版本、路徑、設備型號或參數不同，需要人確認差異
C 不適用：至少一條前提不成立
```

**Q4 要不要自動執行**：用規則不用模型——Q3 = A 且 SOP `reversible: true` 且 Q1 校準後信心 ≥ 門檻 → 自動；其餘進 Keep incident-candidate 讓人按。

**淘汰賽**：SOP 目錄先用 `layer`、`device.model`、`app` 過濾；剩餘 > 10 份才走 `route_bench.py` 的分組淘汰賽（目錄順序）。前十六輪量到過濾後通常不需要。

**不確定區**：任一題校準後信心低於門檻 → 該題交人；Q1 選 F → 直接進人工路徑（§7）。

## 6. 模型分工

| 角色 | 模型 | 跑哪裡 | 做什麼 | 不做什麼 |
|---|---|---|---|---|
| 檢索與 novelty | EmbeddingGemma 2（`google/embeddinggemma-2`，釘 `914f7f89…`；`bench/embed_route.py` 的前綴與中心化） | GB10 CPU，常駐 | 卡 → 向量；歷史卡索引（numpy，幾千張夠用）；先結構過濾再餘弦排序取前 3；novelty = 與最近歷史卡的距離 | 不判層、不判 SOP、不當選項排序的依據 |
| 第一關 | Gemma 4 E4B（Q8） | GB10 GPU | Q1、Q3 先答；校準後信心 ≥ 0.999 直接採用，否則送 26B（v3 的級聯） | 不單獨下自動執行的決定 |
| 判斷 | Gemma 4 26B-A4B UD-Q4_K_M，llama-server ≥ b11371，`--swa-full` | GB10 GPU | Q1–Q3 最終答案；四題共用 state 吃前綴快取 | 不生成文字進自動流程；不開思考 |
| 給人的建議 | 26B 開思考（`bench/think_tail.py`，server 不帶 `--reasoning-budget 0`） | 同上，按需 | 只在掉人工時列「可能的原因」三條給人參考，每案 11–19 秒 | 輸出不進案例庫、不算信心 |
| 規則 | 程式 | — | 去重、分組、Q4、可逆判定、版本比對、時間軸排序 | — |

延遲預算：一個事故從開卡到四題答完 ≤ 10 秒（卡生成 2–5 秒是大頭；26B 每次讀取 0.1–0.3 秒）。

## 7. 掉人工與長回來

1. 進人工的條件：Q1 = F、novelty 超門檻、Q2 三個候選都是 C、任一題信心不足、Q3 = B 或 C。
2. 人在 Keep 事故頁看到：卡、EG2 前三名舊案（含當時的根因與 SOP）、26B 的答案與信心、思考模式列的可能原因（標「建議」）。
3. 人填 `human.{layer, layer_detail, root_cause, sop, outcome}`；需要時新增子類（改 §3 的表與 Q1 判準，版本化）或新增 SOP（§8 格式）。
4. **存回**：卡 + human 欄位進案例庫與 EG2 索引；SOP 進目錄。不訓練。下一次另一台 IPC 出同樣的事，Q2 就會找到。
5. 人的每一次確認、改判、拒絕，都是校準資料：每累積 50 案重算一次溫度與門檻，更新 lock 檔。

冷啟動：前幾個月幾乎全部掉人工，**這是正常的**。可以先把現有 runbook 轉成 SOP 目錄；案例卡只能從真實事故累積。同構 IPC 越多，累積越快。

## 8. 生態系與相容性

**採集（OTel）**
- 一律照 OTel semantic conventions 打 resource attributes：`host.name`、`service.name`、`service.version`、`k8s.namespace.name`、`k8s.pod.name`、`k8s.container.name`、`deployment.environment`，加自訂 `ipc.model`、`ipc.site`。卡生成器的查詢只認這些欄位，欄位穩定查詢才穩定。
- Collector 用 **Grafana Alloy**（OTel Collector 相容，Promtail 已停更）或 OTel Collector contrib 都行；需要的 receiver：`hostmetrics`、`journald`、`filelog`、`kubeletstats`、`k8s_cluster`、`otlp`（app 的 trace／log）；processor：`k8sattributes`、`resourcedetection`。
- 業務指標也走 OTLP（app 自己打 metric），不要另闢一條路。

**microk8s 注意事項**
- snap 自動 refresh 會重啟 microk8s；`snap refresh --hold` 或排定 refresh 時段，並把 `snapd` 的 journald 收進 Loki（K 層子類「snap refresh 重啟」靠它）。
- microk8s 有 observability addon（kube-prometheus-stack、Loki、Tempo），但一群 IPC 建議**集中一套 LGTM**，IPC 上只跑 Alloy 往外送；版本自己控。
- 單節點 dqlite 與憑證過期是 microk8s 特有的子類，查詢清單要有。

**LGTM**
- Loki：LogQL 取模板用內建 pattern 偵測（Drain 類）；卡上放模板不放原始行。
- Mimir：PromQL；kube-state-metrics、node_exporter、smartctl_exporter 都要有。
- Tempo：TraceQL 取錯誤率與延遲；**service graph metrics 是現成的服務依賴圖**，A 層的 `deps_failing` 與 Q1 判準「依賴指向 K 層 NotReady 的服務」靠它，不需要 Keep 企業版的拓撲功能。
- Grafana Alerting：alert 走 Alertmanager 相容的 webhook 進 Keep；alert state history 開起來（可寫 Loki），是重放評估的資料來源。

**Keep（開源版就夠）**
- 去重：partial dedup，指紋欄位用 `host.name` + `alertname` + `k8s.namespace.name`。
- 關聯：規則（同 `host.name` 在 W 分鐘內的 alert 併成一個事故）；企業版的 AI／拓撲關聯**不需要**，Q2 和 Tempo service graph 取代它。
- Workflow：事故建立時 HTTP 呼叫判層服務；回傳寫成 enrichment 欄位：`cbr_layer`、`cbr_layer_conf`、`cbr_similar_case`、`cbr_similar_verdict`、`cbr_sop`、`cbr_sop_verdict`、`cbr_novelty`、`cbr_route`（auto／human）。
- `cbr_route = human` 的開成 incident-candidate；人工填的根因與 SOP 從事故的 resolution 欄位讀回案例庫。
- 相容性原則：判層服務對 Keep 只是一個 HTTP endpoint，**換掉 Keep 不影響服務**；對 LGTM 只用標準查詢 API。

**SOP 目錄格式**（git 管理，Markdown + front matter）
```yaml
id: SOP-O-003
title: /var 磁碟滿導致 postgres 寫入失敗
layer: O
layer_detail: disk_full
preconditions:                 # Q3 逐條對照
  - "layers.O.disk_avail_pct <= 5"
  - "layers.A.deps_failing 含 postgres 或 layers.K.pods_not_ready 含 postgres-*"
applies_to: {ubuntu: ["22.04", "24.04"], microk8s: ["1.30", "1.31", "1.32"]}
reversible: true
steps: [...]
```

**案例庫**：SQLite（卡與 human 欄位）+ `cases.npz`（EG2 向量）放 GB10；備份不出廠。

## 9. 資料與治理

- 真實卡、主機名、IP 都留在 GB10；repo 只進 schema、查詢清單、程式、彙總數字與 lock 檔。
- 評估資料是**歷史事故重放**：拿過去的 alert state history 與 Keep 事故，用 `card_builder` 回溯生成卡，人補 `human` 欄位。沒有歷史事故就只能從 Phase 0 開始累積，不要用合成 storm 算準確率。
- 每張卡要有兩個人獨立判層的紀錄（至少 Phase 0 的 30 張），人都不一致的題模型也不會準。

## 10. 階段與門檻（現在寫死）

| Phase | 做什麼 | 門檻（過／不過） | 不用模型？ |
|---|---|---|---|
| **P0 定義**（2–4 週） | 分層與子類定稿；卡 schema 定稿；手填 ≥ 30 張歷史卡；SOP 目錄 ≥ 10 份轉成 §8 格式 | 兩人獨立判層一致率 ≥ 0.85（30 張）；每張卡 `missing` 可填 | 是 |
| **P1 卡生成器** | `cbr/queries.yaml` + `card_builder`，對 ≥ 20 個歷史事故回溯生成卡 | 自動卡包含人判定的關鍵證據 ≥ 80%（人逐張勾）；`missing` 正確率 ≥ 90% | 是 |
| **P2 離線判斷**（≥ 50 張有 human 欄位的卡） | Q1–Q3 重放；EG2 檢索；novelty；E4B 級聯；校準、lock 檔 | Q1 校準後 ε=5% 下 coverage ≥ 0.7 且實際錯 ≤ 5%；Q2 EG2 前三命中 ≥ 0.85 **且**贏同一張卡的 TF-IDF ≥ 0.05（贏不了就用 TF-IDF）；novelty 在誤報 10% 下抓到真正新案 ≥ 0.8；Q3 的 A 誤判（不該套說可套）≤ 2%；E4B 處理 ≥ 30% 且整體準確率差 ≤ 0.01 | 否 |
| **P3 影子模式**（4 週） | 接進 Keep，只建議不執行 | 建議與人最終判定一致 ≥ 0.85；掉人工比例與其中「真的新案」比例報出來 | 否 |
| **P4 自動**（可逆 SOP） | Q4 規則放行 | 自動執行錯誤率 ≤ 2%（錯誤預算），超過即關回影子模式 | 否 |

每個 Phase 的門檻沒過就停在那個 Phase 修，不往下走。P2 樣本 < 100 張時，結論只寫個數與區間，不寫百分比定論。

## 11. 評估設計與對照組

對照組少一個都不算數：
1. **純規則**：最早異常的層就是根因（Q1）；同 `host.name` + 同 alertname 集合就是同一件事（Q2）。
2. **TF-IDF**：同一張卡的文字算餘弦當檢索（Q2 候選）。模板文字上它可能跟 EG2 一樣好。
3. **只用 E4B**：不送 26B。
4. **給原始日誌不給卡**：確認卡的價值不是模型的價值。

指標：Q1 準確率與選擇性準確率；Q2 前三命中、novelty 的 ROC；Q3 準確率與 A 的誤判率；每案讀取次數與延遲；掉人工比例。統計：McNemar（精確二項），樣本不足就報個數。

## 12. 風險與不要做

- **固定查詢看不到的東西**：清單外的異常不會在卡上，模型會很有把握地選錯層（v14 教訓：模型對名單外的東西不會猶豫）。對策：`missing` 欄位 + Q1 的 F 選項 + 人工路徑永遠在。
- **版本漂移**：microk8s 或 app 升版後舊 SOP 可能失效；Q3 的 B 選項與 `applies_to` 處理，卡上一定有版本。
- **自動執行**：只做可逆的；每一次自動執行寫稽核紀錄；錯誤預算超過就自動關。
- **過度相信相似**：Q2 的 A 要求關鍵證據一致，不只是 alert 名字一樣；同徵狀不同原因是 B。
- 不要做：訓練或微調任何模型；讓模型生成文字進自動流程；用合成資料算準確率；在 alert 層級塞模型；買 Keep 企業版的 AI 關聯（Q2 + service graph 取代）；開思考進自動流程；把 EG2 當選擇器。

## 13. 要人決定的事

1. IPC 有幾台、幾種型號；過去一年有多少次事故留有紀錄（決定 P2 何時有 50 張卡）。
2. LGTM 是集中一套還是每台一套；Alloy 還是 OTel Collector。
3. 哪些 SOP 可以算「可逆」。
4. Keep 要不要用；不用的話 Grafana OnCall／Alertmanager webhook 直接打判層服務也行，§8 的欄位對應要改。
5. 人工路徑的介面：Keep 事故頁、LINE 群、還是工單系統。

## 14. 結論（做完填）

（待填：P0–P4 各自照 §10 的門檻填過／不過，預測錯了照實寫。）
