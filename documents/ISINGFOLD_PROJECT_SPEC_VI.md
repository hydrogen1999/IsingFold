---
title: "Đặc tả Dự án IsingFold"
subtitle: "Minor embedding ưu tiên chất lượng với validity chính xác, reinforcement learning trên typed graph và evaluation có xác thực"
author: "Dự án IsingFold"
date: "12 tháng 9 năm 2026"
lang: vi-VN
---

# Mục tiêu và luận điểm nghiên cứu

IsingFold xem minor embedding là một bài toán quyết định tuần tự có xét chất lượng. Một hệ thống hữu
ích phải trả về embedding hợp lệ, giữ đủ khả năng kết nối để sửa các trạng thái khó, và cải thiện
chất lượng nghiệm Ising sau khi lập trình và lấy mẫu. Chỉ giảm số physical qubit hoặc độ dài chain
không phải là mục tiêu cuối.

Thí nghiệm chính dùng Profile I. Một classical initializer đã đăng ký tạo embedding khả thi khi có
thể. Learned policy sau đó thực hiện các phép rewrite nhiều chain có giới hạn, trong đó cho phép
overlap tạm thời có kiểm soát, còn protected archive giữ một phương án hợp lệ dự phòng. Independent
validator là thành phần duy nhất chấp nhận embedding cuối. Construction từ trạng thái rỗng là một
profile riêng, có phân phối khởi tạo và failure constraint khác.

Giả thuyết trung tâm phải được đo trực tiếp:

> Trên population đã niêm phong, cùng deployment selector, giới hạn tài nguyên rõ ràng và cùng
> online wall-clock envelope, chuỗi learned rewrite cải thiện failure-aware downstream Ising utility
> so với same-support control và hệ stock minorminer được tune chỉ trên validation.

Đây là giả thuyết, không phải hệ quả mặc nhiên của kiến trúc. Muốn khẳng định learned model tốt hơn
phải dùng held-out evaluation ba seed theo đúng registry.

# Objective và validity

Với logical Ising instance $I=(G_L,h,J)$,

$$
E_I(s)=\sum_i h_i s_i+\sum_{(i,j)\in E_L}J_{ij}s_i s_j,
\qquad s_i\in\{-1,+1\}.
$$

Một embedding gán mỗi logical variable $i$ vào một branch set $C_i$ khác rỗng và liên thông trên
realized active hardware graph. Embedding được trả về phải thỏa tất cả điều kiện sau:

1. mọi branch set khác rỗng, liên thông và chỉ dùng phần cứng đang hoạt động;
2. các branch set đôi một rời nhau tại thời điểm return;
3. mọi logical coupling khác zero có ít nhất một physical contact giữa hai branch set;
4. physical-qubit cap và mọi prospective work cap đều được thỏa;
5. chương trình vật lý bảo toàn tổng field và coupling logic theo compiler đã đăng ký.

Search state có thể chưa đầy đủ hoặc còn overlap trong envelope O1. Search legality,
embeddability, return validity và terminal success là bốn predicate khác nhau. Neural model chỉ
được xếp hạng những action mà environment đã materialize và đánh dấu legal. Dự đoán của model không
thể tự tạo ra validity.

Endpoint đăng ký là IF-Q3-S0. Một program-feature selector đã đóng băng chọn đúng một strength trong
$(0.5,1,2,4)S_I$ mà không quan sát ground energy, planted solution, sample count hoặc evaluator
result. Với terminal embedding hợp lệ $\phi$, selected-strength quality là

$$
Y_\psi(\phi)=p_{j_\psi(\phi)}(\phi),
$$

trong đó $p_j$ là xác suất majority decoding đạt certified logical ground energy trong tolerance
đã đăng ký. Failure-aware utility là

$$
U=A\,Y_\psi(\phi),
$$

với $A=1$ chỉ khi embedding trả về vượt qua fresh independent validation. Các failure thông thường
ở initializer, search, budget, timeout, compilation hoặc invalid return vẫn nằm trong mẫu số với
$U=0$. Integrity failure như artifact hỏng hoặc thiếu read sẽ làm thí nghiệm dừng, không bị đổi
thành utility zero.

Physical-qubit count, maximum chain length, connectivity, native work, latency, memory và read count
là hard constraint, Pareto axis hoặc explanatory metric. Dùng thêm qubit đôi khi có thể cải thiện
sampling landscape, nhưng số qubit không bao giờ tự tạo positive reward.

# Kiến trúc hệ thống

Runtime tách learned preference khỏi exact mechanics.

| Lớp | Trách nhiệm | Shortcut bị cấm |
|---|---|---|
| Data authority | Xác thực CandidateBank, lineage design, evaluator target và ground certificate | Dùng self-signed corpus làm scientific authority |
| Classical initializer | Tạo embedding khởi đầu của Profile I trong prospective work limit | Restart không tính phí hoặc fallback ẩn |
| Exact environment | Quản lý state, candidate materialization, action mask, overlap, archive, transition và work | Learned validity hoặc sinh candidate sau khi đã xem outcome |
| IF-Core | Mã hóa graph và bound action, rồi sinh masked policy và critic | Truy cập witness, optimum energy hoặc evaluator outcome |
| Strength selector | Chọn một terminal program strength từ deployment-visible feature | Dùng oracle bốn strength khi deployment |
| Independent evaluator | Compile, sample, decode và chấm fresh terminal read | Tái dùng training read cho confirmation |
| Evidence layer | Gắn raw row, sidecar, receipt, identity và aggregate rule | Lọc chỉ success hoặc dùng mutable result folder |

Bốn giai đoạn cổ điển ordering, placement, routing và repair đều được biểu diễn. Ordering xuất hiện
trong proposal priority và construction action. Placement và routing được mở trong construction
profile. Repair và refinement là các action family chính của Profile I. Learned policy thay thế
quyết định preference trong luồng này, còn exact graph check và programming check luôn nằm ngoài
model.

## State, action và work

Một decision state chứa logical instance, active host, immutable objective context, relaxed branch
set, đầy đủ ownership claim, valid archive, search memory, vector work chín tọa độ còn lại, và một
candidate batch đã materialize hoàn chỉnh. Action bind cả membership OLD lẫn NEW. Group rewrite
được áp dụng nguyên tử để một coordinated move hợp lệ không bị loại chỉ vì thứ tự tuần tự tùy ý.

Finite action grammar gồm `PLACE`, `ROUTE`, `REWRITE`, `GROUP_REWRITE`, `REPAIR`, `RESTART`,
`COMMIT` và stop behavior đã đăng ký. Profile I chủ yếu dùng rewrite, repair, restart, restore và
commit. Action cap gồm 64 state-changing candidate, tám archive commit và một optional stop. Archive
gồm một protected initializer và bảy entry FIFO.

Candidate construction không dùng outcome. Nó trộn route length, occupancy pressure, contact
multiplicity, local free neighborhood, bottleneck feature và fixed random perturbation. Mọi attempted
route, rejected proposal, materialization, validation, archive operation và optional cut feature đều
được tính phí trước khi actor chọn. C++ kiểm tra từng prospective work delta và trả về exact
non-exceeding ledger cùng tọa độ làm cạn budget.

# Kiến trúc model IF-Core

IF-Core là typed graph actor-critic với hidden width 128. Kích thước này đủ gọn để thực hiện các
controlled comparison.

1. Ba dual local block xử lý riêng logical graph và hardware graph.
2. Hai typed fusion block trao đổi thông tin qua ownership relation và conflict relation.
3. Phase-aware factor encoder giữ riêng identity của chain OLD, NEW và ARCHIVE.
4. Gated one-dimensional convolution hai layer mã hóa route có thứ tự.
5. Action encoder kết hợp descriptor, chain factor, route, conflict và archive reference.
6. Segmented mean/max pooling tạo state summary và action-set summary.
7. Masked categorical actor chỉ chấm exact support.
8. Utility critic ước lượng failure-aware terminal return. Failure critic riêng được bật trong
   construction experiment.

Core tensor width được cố định: logical node có 20 value slot cộng knownness bit, hardware node có
18 cộng knownness bit, global context có 32 cộng knownness bit, và action descriptor có 24 cộng
knownness bit cùng opcode encoding. Implementation batch graph trunk thành một disjoint union rồi
tách lại từng observation trước các phép factor, route, archive, action và categorical reduction.
Environment collection vẫn tuần tự vì support kế tiếp phụ thuộc action vừa chọn.

Model ladder gồm IF-MLP, IF-Dual và IF-Core. IF-MLP kiểm tra liệu engineered action feature đã đủ
hay chưa. IF-Dual kiểm tra logical/hardware message passing tách biệt mà chưa có typed fusion đầy đủ.
IF-Core kiểm tra relational structure của ownership, conflict và joint action. Optional cut token,
global relay attention, recurrent memory và learned proposal generation phải có mechanism cell riêng,
không được đưa vào ngầm.

# Learning objective và optimization

Quality row v7 ước lượng $Q_U^\mu(s,a)$ dưới frozen continuation policy $\mu$ đã đặt tên. Semantic
target là `if-q3-s0-qmu-7`; corpus publication, shard, merge và preflight envelope có schema version
riêng. Mỗi row giữ complete action support, exact observation, action inclusion propensity,
continuation seed, terminal outcome và independent evaluator count. Mọi sampled action đều huấn
luyện bounded $Q^\mu$ head. Protected incumbent COMMIT, nếu có, luôn được chọn và còn tạo
within-state delta target. Supervised ranking dùng simultaneous plausible-best set $B_s$, không ép
một strict winner khi finite-read uncertainty chưa phân giải được tie:

$$
L_{\mathrm{sup}}
=-\frac{1}{|\mathcal S|}\sum_s
\log\frac{\sum_{a\in B_s}\exp\ell_a}{\sum_{a\in E_s}\exp\ell_a}.
$$

Warm start đồng thời khởi tạo utility critic bằng chính các train row đã được authenticate. Exact
Horvitz-Thompson reduction dùng propensity đã lưu để ước lượng policy chọn action exact-legal đầu
tiên theo phân phối đều rồi tiếp tục bằng $\mu$. Continuation count tác động tới within-state
regression weight và bounded-variance critic confidence weight, nhưng không âm thầm đổi target
action distribution. Khi mới đánh giá một phần action support, sampling uncertainty vẫn khác
không, do đó lặp thêm continuation trên đúng subset đó không tạo confidence vô hạn. Bốn supervised
term dùng corpus-level denominator đã khóa. Các memory minibatch tích lũy thành đúng một
full-corpus gradient trước mỗi optimizer step; paper grid khóa 200 step như vậy. Target chỉ được tạo
sau model forward và không đi vào observation, mask, candidate, logit hay critic input. Sau đó PPO
fit lại critic theo learned actor bằng fresh on-policy episode.

PPO dùng complete episode, generalized advantage estimation, exact action-support replay và
episode-sum reduction. Với ratio $r_t(\theta)$ và advantage $\widehat A_t$,

$$
L_{\mathrm{clip}}
=-\mathbb E\!\left[
\min\left(r_t\widehat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t\right)
\right].
$$

Joint loss cộng utility-value error, construction failure-value term khi được bật, và entropy
schedule đã đăng ký. Legal categorical entropy được chia cho log của số legal action; singleton
support có entropy bằng không. Coefficient giảm từ 0.01 đến floor dương 0.001. Dropout cùng
stochastic tensor augmentation bị tắt trong PPO để unchanged policy replay đúng likelihood ratio
một. Gradient không đi qua old behavior output, environment validity, proposal generation hoặc
frozen selector.

Reference value gồm horizon 32 decision, 64 complete episode mỗi rollout, bốn PPO epoch,
minibatch 256 transition, $\gamma=1$, GAE $\lambda=0.95$, clip 0.2, AdamW learning rate
$3\times10^{-4}$, gradient cap 0.5, full-buffer KL soft target 0.01 và hard limit 0.02. Mỗi
tentative epoch snapshot model, optimizer và RNG state. Khi vượt hard limit, toàn bộ epoch được
rollback rồi retry với learning rate giảm một nửa, tối đa ba lần. Nếu hết retry, last safe boundary
được giữ nguyên; khi đạt soft target thì dừng các epoch còn lại. Ba scientific seed là 1103, 2207
và 3301.

KL rollback và normalized entropy chỉ thay đổi độ ổn định của quá trình training. Chúng không thay
đổi terminal reward, IF-Q3-S0 strength selection, data split, target label, deployment action
selection hoặc publication endpoint. Mọi thay đổi khoa học như vậy phải thuộc một experiment được
đăng ký riêng.

# Data preparation và authority

EmbedBench là một package độc lập. IsingFold chỉ dùng authenticated CandidateBank-v2 export, không
fork generator vào training runtime. Export kết hợp feasible ink-drop witness trên Chimera, Pegasus
và Zephyr, realized fault map, exact structural motif, planted frustrated-loop Ising instance,
application-derived instance và replayable quality record. Embedding witness chỉ chứng nhận
feasibility. Planted logical certificate chỉ thiết lập ground energy. Không loại nào xác định
quality-optimal embedding.

Production importer còn yêu cầu corpus-design manifest v2 được pin độc lập. Nó xuất prepared
schema v4 và kiểm tra:

- exact quota trên immutable base lineage;
- ít nhất 1.024 base lineage cho train, 512 cho validation và 1.546 cho sealed test;
- cả application-derived lẫn synthetic origin, nhiều host family và faulted condition;
- ba trục riêng cho embedding difficulty, sampling difficulty và decision-quality difficulty;
- hard/OOD coverage được cố định mà không dùng outcome của phương pháp;
- confirmatory power/precision target bao phủ toàn bộ realized test population;
- validation-only baseline-tuning target bao phủ toàn bộ realized validation population.

Thiết kế tuning trên validation yêu cầu ít nhất 128 independent lineage. Phép tính paired valid
return đã đăng ký cho kết quả 127 với one-sided alpha 0.05, power 0.8, discordance 0.1, true
difference 0.05 và margin 0.02. Phép tính precision của bounded paired utility cho kết quả 97 với
two-sided 95 percent half-width 0.2. Cả hai đều được làm tròn lên prospective floor 128.

Prepared v4 tách vật lý evaluator target thành ba file `train`, `val` và `test`; public manifest chỉ
chứa hash, count và set digest. Mọi command mở target đều cần publisher ID và attestation record
digest từ kênh ngoài. Publisher attestation v2 bind prepared manifest, partitioned target authority
và các evidence manifest có hash riêng. `verify-ground-certificates` chạy standalone verifier được
pin riêng theo protocol v2, ghi ba partition receipt trước rồi mới publish `root.json`. Root chỉ chứa
commitment và census metadata nên load root không thể mở target. Downstream stage chỉ mở đúng một
partition đã được authorize và giữ cả `TargetAccessReceipt` lẫn ground-partition receipt.

Selector label và quality label là hai loại khác nhau. Selector data chứa four-strength count block
đầy đủ trên train và calibration partition. Quality-v7 data chứa exact-replay action
counterfactual. Quality job dài dùng deterministic whole-lineage shard. Merger nhận mỗi registered
shard đúng một lần, loại gap và overlap, replay toàn bộ row, tính lại denominator và xuất canonical
corpus. Quality preflight yêu cầu tối thiểu 128 resolved row và 128 resolved independent lineage.
Không shard đơn lẻ nào được đi vào training.

# Model selection và training

Bốn gate phải vượt qua trước khi chọn architecture:

1. exact conformance của graph, program, structural label và certificate;
2. candidate support có quality headroom đo được;
3. frozen strength selector phân biệt được strength và hơn fixed-strength/random control;
4. đủ valid return và utility signal không bão hòa.

Main grid có đúng (9+18) cell.

| Giai đoạn | Family và method | Seed | Cell |
|---|---|---:|---:|
| Representation | IF-MLP, IF-Dual, IF-Core với supervised training | 3 | 9 |
| RL value | selected simpler family và IF-Core, kết hợp supervised-only, PPO warm-start, PPO from scratch | 3 | 18 |

Validation selection cho mỗi registered seed và immutable base lineage trọng số bằng nhau, áp dụng
valid-return noninferiority gate so với `return_initial`, rồi xếp survivor theo unconditional
IF-Q3-S0 utility và online cost. Test partition không được tham gia. Capacity control nằm ngoài main
grid: nếu IF-Dual được chọn làm simpler family, IF-Dual width 128 với chín local block được so với
IF-Core trên đủ ba seed, với chênh lệch parameter count không quá 0,1 percent.

PPO không dùng hand-authored CandidateBank incumbent như thể đó là deployment sample. Với mỗi
training seed, một target-free initializer-bank plan cố định lineage-equal episode schedule, public
task identity, exact LAC runtime và complete-system config. Mỗi conditional episode có một chuỗi hữu
hạn các initializer draw xác định. Generation dừng ở lần thành công đầu tiên, vẫn hạch toán work của
các draw thất bại, và fail closed nếu hết draw cap. Bank chỉ được seal khi mọi conditional episode
đã resolve. PPO chỉ gắn train target sau khi bank được xác thực. Vì vậy training distribution của
policy được khai báo rõ là conditional trên deployment initializer thành công, còn final estimand
vẫn giữ initializer failure trong denominator.

Sau khi RL-value receipt được đóng băng, complete-system confirmation train mới cả ba seed, không
resume. Evaluation bắt đầu từ mọi sealed task và repetition trước initializer. Raw terminal
evidence, outcome projection, runtime source digest, model identity, selector identity, quality
authority, target-access receipt, ground-partition receipt và work ledger đều phải được xác thực
trước khi tính crossed seed-by-lineage interval.

# Baseline và evaluation

Same-support control gồm `return_initial`, random masked choice, resource-lexicographic choice,
classical quality-aware ranking, IF-MLP, IF-Dual, supervised IF-Core và các PPO variant. Chúng tách
proposal headroom, representation value và sequential-policy value.

Stock minorminer là một whole-system arm riêng. Finite registry chứa default được viết rõ và các
variant time-saturating xếp hạng theo resource hoặc frozen-selector quality. Mọi candidate chạy trên
toàn bộ validation population ở cả ba registered seed, với một machine identity cho mỗi seed. Một
immutable receipt đóng băng candidate, không chọn seed thuận lợi, trước khi mở test. Test arm dùng
cùng sealed pre-initialization census, pair key, seed schedule, selector, evaluator read,
prospective cap và total online wall-clock envelope với learned arm. Solver-specific internal work
coordinate được công bố, không bị tuyên bố là bằng nhau.

Publication comparison dùng fixed-sequence familywise rule. Nó kiểm định valid-return
noninferiority với margin 0.02 trước. Utility superiority chỉ là confirmatory khi noninferiority
đạt, paired unconditional IF-Q3-S0 lower confidence bound lớn hơn zero, và sample-size/precision
guard đạt. Conditional quality, qubit use, chain length, connectivity, work, latency, memory và
failure reason là secondary analysis.

Các complete-system evaluation và stock tuning dài dùng chung một resumable protocol cho ba
workflow: learned confirmation, tuned-stock confirmation và external validation tuning. Plan commit
toàn bộ public population, run coordinate, seed derivation, execution contract và lineage shard
trước khi mở target partition. Mỗi shard sở hữu trọn immutable base lineage và giữ nguyên
identity-derived seed như khi chạy không shard. Shard receipt chứa partition target access, ground
authority, compute class, actual node provenance, raw receipt, outcome và terminal evidence. Merger
yêu cầu complete nonoverlapping key census và tự tính lại unsharded sufficient statistics.
Publication run dùng `pinned-venv` trên Apollo hoặc `apptainer` trên Goose; `bare-metal` chỉ dành cho
diagnostic.

Final four-strength diagnostic được niêm phong sau khi hai arm và cả ba seed đã đóng băng. Config
đăng ký là `configs/final_strength_audit_v1.json`; registry pin cả record digest và file SHA-256.
Nó cố định tối thiểu 128 base lineage, tối đa 16 lineage mỗi signed stratum, 4.096 read mỗi block,
20.000 bootstrap
replicate và 128 shard. Outcome-blind plan được niêm phong trước test outcome. Sáu source run được
xác thực trước khi execution manifest được niêm phong, và manifest này được niêm phong trước khi
bắt đầu audit read. Với mỗi valid return, block A lấy mẫu mới trên
cả bốn strength và chọn empirical oracle với lowest-index tie rule. Block B lại lấy mẫu mới trên cả
bốn strength và ước lượng oracle-minus-deployed regret. Invalid return vẫn nằm trong coverage và
worst-case sensitivity denominator. Deterministic shard phân hoạch complete opportunity key; merger
yêu cầu full nonoverlapping census, xác thực lại source evidence, phục hồi canonical order, tính lại
summary và publish nguyên tử. Audit này không thể thay đổi training, model selection, baseline
tuning hoặc primary endpoint, và không phải primary evidence cho learned-method superiority.

# Reproducibility và HPC

Compatibility chain gồm:

1. prepared-v4, publisher attestation v2 và ground-certificate protocol v2 với target-free root cùng
   partition receipt;
2. selector-label manifest v4 và selector bundle v4;
3. quality record v7, shard/merge v5 và quality preflight v2;
4. release gate v6, representation selection v4 và RL-value freeze v4;
5. checkpoint v3 và training-run receipt v4;
6. learned complete-system receipt v3, evaluation report v4 và aggregate v2;
7. external tuning run/selection v2, external complete-system receipt v3/report v4 và paired
   aggregate v4;
8. resumable evaluation plan v2, shard receipt v3 và merge receipt v2;
9. final-strength config v1, sampler identity v1, plan v2, row v1, execution manifest v2, shard v1,
   merge v1 và audit receipt v2.

Mọi scientific output là canonical finite JSON, có self-digest, được ghi nguyên tử, immutable theo
mặc định, và được kiểm tra với caller-supplied file pin khi trust boundary yêu cầu. Runtime identity
gồm loaded native-extension byte và imported Python source.

Apollo không có Slurm nên launcher chạy trực tiếp. Mọi computation trên Goose chỉ chạy qua Slurm và
dùng `srun` bên trong allocation. Seed-to-host assignment giữ nguyên giữa training và validation.
Distributed initializer generation, labeling, evaluation và auditing phân hoạch immutable whole
key, không cắt một evaluator block. Publication evaluation bind cluster, scheduler, Slurm partition
khi áp dụng, CPU/GPU identity, thread count, deterministic setting, runtime hoặc environment image,
và source digest.

# Giới hạn của claim

Kiến trúc thiết lập exact interface và falsifiable test. Nó không tự chứng minh checkpoint đã train
tốt, learned method thắng minorminer, simulator gain chuyển được lên QPU, hoặc IsingFold thắng mọi
graph family. Claim chỉ được nâng lên sau registered gate, fresh three-seed confirmation, complete
failure-aware denominator, strong baseline và independently auditable receipt.

Phương trình model và tensor slot chi tiết nằm trong model specification. Exact execution command
nằm trong training operations guide. Modular architecture nằm trong thư mục
`IsingFold_Architecture_Rev2`, còn stock-baseline tuning có specification riêng trong thư mục này.
