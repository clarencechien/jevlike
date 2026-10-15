# data/v21/kikemu — exp5 晶晶體語料的正解與既有 ASR 輸出（v21 抓錯閘門用）

來源：clarencechien/kikemu @ `1a9cea4`（2026-10-15），唯讀複製，一個位元都不改。音訊不在這裡（kikemu 慣例：不散布），用 kikemu 的 `exp5/scripts/prep_audio.py` + `exp5/scripts/degrade.py` 重建，sha256 必須與 `audio_manifest.json` 相符。

| 檔 | 用途 |
|---|---|
| `reference/T1..T3.txt` | 正解（兩路 Gemini 合併 + `overrides.json` 人工裁決；本輪當 oracle） |
| `verify/*.gm35.txt`、`*.gm36.txt`、`*.sm.txt` | 三路獨立轉寫（gm36 用來量正解的雜訊地板） |
| `terms.json` | 每段的英文術語與變體（召回判定凍結在 kikemu `make_terms.py`） |
| `audio_manifest.json`、`picks.json`、`degrade_meta.json` | 音檔指紋、來源視窗、聲學條件 |
| `raw/Xbrz_gpu/*.json` | Breeze ASR 25（T4 fp16，revision cffe7ccb）的逐字稿與時間戳段落，12 檔 |
| `raw/Xgma_e4b_greedy/*.json` | Gemma 4 E4B 自己當 ASR（貪婪，M0／M3）的逐字稿，6 檔；D5 雙 ASR 分歧用 |

| 路徑 | bytes | sha256 |
|---|---|---|
| `audio_manifest.json` | 2414 | `d1a1e9140c673b029f61dbb92a5cc8b10f2957752f4c7165184d68a4acd2c8e4` |
| `degrade_meta.json` | 141 | `ea169a663336b6e04ba44cf67700b74d71e8c2cd8f4cf9d9b9a93c009f84eb01` |
| `picks.json` | 15313 | `24ea801af535a6c4ed684cfba87b2c2893be530b91b57dbe5cd5305921180b79` |
| `raw/Xbrz_gpu/T1__M0.json` | 18595 | `0516b4d71109c64ef25e722644c78a3d1391b2064f33c9eec3f1e722c5f4dcd7` |
| `raw/Xbrz_gpu/T1__M1.json` | 19075 | `7fe5cd07ffb8c967272ec4deef0b1f7a575e65514eff6addd2e1959cf73bc0e4` |
| `raw/Xbrz_gpu/T1__M2.json` | 18975 | `fc24501640c104de060b57a20e5b04e8e3922e8d1f6204463ca8f6a82c380c61` |
| `raw/Xbrz_gpu/T1__M3.json` | 17864 | `f1c3f78e03606323adfd0cd986b19ee7f24433620254c3ce9103dc9592636ce7` |
| `raw/Xbrz_gpu/T2__M0.json` | 15780 | `8f5777790bc7443ede6e2e88f028ec78245579d0540be9836f35b74fa3eda203` |
| `raw/Xbrz_gpu/T2__M1.json` | 16124 | `87cb648008148395fdf19aca167acfa31c4741b8697952cd245f75805749dd72` |
| `raw/Xbrz_gpu/T2__M2.json` | 16268 | `d00144bbb2bbdb56c7ec341ffda2ad66bcb49a7e455bbc5727a14d3dc9dbe216` |
| `raw/Xbrz_gpu/T2__M3.json` | 16056 | `83654e25b8921e2084cf117fb3d38a2ca76c76aac19cc1c4a2722c28c64a180b` |
| `raw/Xbrz_gpu/T3__M0.json` | 15667 | `016044e583678bca2a99f5b3e88dcfb2a6c5b9f6d0f777b05c69c9a7db5104b2` |
| `raw/Xbrz_gpu/T3__M1.json` | 15804 | `b6ff1b020098a844e1c3faf41425410b4faf932cf80c31dcb180f292342359f9` |
| `raw/Xbrz_gpu/T3__M2.json` | 16057 | `2755a237dc986cd3f973d7d1c7b6a9c0e4b2996195ce434c4b1295adf0775996` |
| `raw/Xbrz_gpu/T3__M3.json` | 15769 | `abd33e613aaa976666389571d5bd16ee4d5f86d34fa24ba91501620174803462` |
| `raw/Xbrz_gpu/_meta.json` | 460 | `ee68ca4e9cd8f620574901b3d0f2f2c8023890375b0cba5bb920f3b7343fd55f` |
| `raw/Xgma_e4b_greedy/T1__M0.json` | 6443 | `17593918a5036e653b57a7b073fb67b476455129f7f95eca782857ab60c5f1d2` |
| `raw/Xgma_e4b_greedy/T1__M1.json` | 6338 | `d0ac43ae7f8b6f0f6b9c420845467e9f8495d715ca195e6a5cdeafeb19c924c3` |
| `raw/Xgma_e4b_greedy/T1__M2.json` | 6343 | `382f3c4c833dd04558708927c25703217866b7f6e958e8c553c9d19e0d0c18f9` |
| `raw/Xgma_e4b_greedy/T1__M3.json` | 6037 | `ff28a3e377cff42d124599f08478fbf4ae70e01e123b890cf2a191f6f81c8b48` |
| `raw/Xgma_e4b_greedy/T2__M0.json` | 5920 | `219954d9e5a3abc537235702c674a59f6047e4db724c1a02451b10cd8954276c` |
| `raw/Xgma_e4b_greedy/T2__M1.json` | 5861 | `7e45416b2ca893b6ca880cd3358599fffe735aa53949d3074fad3467260873d4` |
| `raw/Xgma_e4b_greedy/T2__M2.json` | 5900 | `cc961bc724dbdb37a6eb74eee265505f6823afa2269787e8293206183b7c162c` |
| `raw/Xgma_e4b_greedy/T2__M3.json` | 5695 | `90fab1ac96c926b99a7880eeedbe816fd017b860502f9d8fcb9a2437a793156e` |
| `raw/Xgma_e4b_greedy/T3__M0.json` | 5755 | `3c25fb7730676a73ffb5117804419887ecc7ba7155b2bef5d3d2d8a4bd42fb6f` |
| `raw/Xgma_e4b_greedy/T3__M1.json` | 5654 | `0176f0ba0b4937c271d00a1fd0b0cdfcf7797e8c9daaaa9566fc2788763666b2` |
| `raw/Xgma_e4b_greedy/T3__M2.json` | 5657 | `b1661a24a4f6974fb238cd64138dd38f1ce7662ac201a6236278c17f947d4a93` |
| `raw/Xgma_e4b_greedy/T3__M3.json` | 5485 | `d4081b6b7129a9e1d6320ab07100939a269155e5d58e58fb227da715afa443a6` |
| `raw/Xgma_e4b_greedy/_meta.json` | 843 | `8c9d5258d53f3b74572f4983d76931c6afce2384658940df6919227e583d4890` |
| `reference/T1.txt` | 5193 | `a2367e07cb65ff8265d22b0078223c715c69c3bd00979f5c87ac794a6572720f` |
| `reference/T2.txt` | 4511 | `c0e28bd7fea27b46f0b9675a7c86896cf4b02c6903b61bb9f4a0d752fc7714f2` |
| `reference/T3.txt` | 4515 | `e5b484f563199093567834ff0de4b43af2f4fdd689b96c18d25112bab7fc249a` |
| `reference/overrides.json` | 1915 | `e61e2c622d613a9abe6ceb257e9d85afb5d8516326da5d433f734caa5131db5f` |
| `terms.json` | 8459 | `f41434030b07855a090b1836235808c5258a7ae2afa80e9b5a187551cd9c06d3` |
| `verify/T1.gm35.txt` | 6050 | `451219c8591e8de68f3a40b0509baba188fc0923d05b0e27049d9325b53ecad0` |
| `verify/T1.gm36.txt` | 5649 | `01f80ddf134ed527558e25a53694a86ced5f10c313c94b4a1dda031c5bc75f3c` |
| `verify/T1.sm.txt` | 5530 | `d08c27c99df43a6069872f660e3ed36cd8a154bc68e2d9c2e7e92b6b8e3c964c` |
| `verify/T2.gm35.txt` | 5172 | `9c421513e2877c7c0f736851064b13c34159763837b72b4dc4b6a2c6356c0b79` |
| `verify/T2.gm36.txt` | 4967 | `a9fb12643a73b144d8c465b514e0d135b5be63ce2df6b4e0db558c0a086b7fa8` |
| `verify/T2.sm.txt` | 5079 | `2b94a57bae4bed5ebf9f3d71dd4e8d8c48684ccd48a87228b800f3843130ca73` |
| `verify/T3.gm35.txt` | 5224 | `3927c5dfae02b4101e41d13639dfeb5976cb358711a7c67f56fa5ab19a0a8972` |
| `verify/T3.gm36.txt` | 5001 | `97b87eaaf3c08a2f35ca65693390cc68f100b92c761407e9c9bdac9dfb46012b` |
| `verify/T3.sm.txt` | 5079 | `2c77db6e292a64025518b8ac51e91a7fd8959148c857bfd3b64a728c98a70410` |
