# Reflection — Lab 19

**Tên:** Lê Trung Kiên
**Cohort:** _<xin điền, vd A20-K1>_
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên 50 golden queries (lite path, `bge-small-en-v1.5`): **hybrid thắng trung
bình** (P@10 = 78.6% vs BM25 77.8%, vector 73.2%), thắng rõ trên `mixed`
(100.0% vs 98.5% vs 97.0%) nhờ RRF gộp tín hiệu từ-khoá + ngữ nghĩa. Trên
`exact`, BM25 = hybrid (96.7%) vì query gần trùng token, không cần trợ giúp.
Bất ngờ nhất: ở `paraphrase`, BM25 (33.3%) nhỉnh hơn cả hybrid (32.0%) và
vector (24.0%) — ngược lý thuyết. Lý do: `bge-small-en` là embedding **tiếng
Anh**, yếu với câu hỏi diễn đạt lại bằng tiếng Việt, semantic score nhiễu kéo
RRF xuống theo — đúng gotcha README đã cảnh báo; đổi `bge-m3`/`multilingual`
sẽ cải thiện.

**Khi nào không dùng hybrid:** pure BM25 khi query exact-match (mã, log, code
search) — vector chỉ thêm nhiễu/latency. Pure vector khi latency-critical,
query luôn là diễn đạt lại, hoặc đã dùng embedding đa ngữ mạnh nên BM25 không
còn đóng góp thêm.

---

## Điều ngạc nhiên nhất khi làm lab này

BM25 thắng vector trên slice `paraphrase` — ngược trực giác thông thường —
nhưng đúng ra là hệ quả trực tiếp của việc chọn embedding tiếng Anh cho corpus
tiếng Việt, không phải lỗi thuật toán RRF.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
