# Massive MIMO TDD – 5G Simulation

## 1. Giới thiệu

Repository lưu trữ mã nguồn và dữ liệu phục vụ đề tài nghiên cứu, mô phỏng và đánh giá hiệu suất phổ của hệ thống Massive MIMO TDD đường lên trong mạng 5G.

Chương trình khảo sát ba kịch bản mô phỏng KB1, KB2 và KB3, đồng thời đánh giá ba phương pháp kết hợp tín hiệu MR, ZF và M-MMSE.

## 2. Danh sách tệp

| Tệp | Chức năng |
|---|---|
| [M_MophongMIM_main_tables.py](./M_MophongMIM_main_tables.py) | Chương trình mô phỏng chính, thực hiện ba kịch bản và xuất kết quả |
| [Phan_Tich_Ket_Qua.ipynb](./Phan_Tich_Ket_Qua.ipynb) | Notebook phân tích và tổng hợp kết quả mô phỏng |
| [plot_massive_mimo_results.m](./plot_massive_mimo_results.m) | Script MATLAB trực quan hóa SE, SINR và so sánh các kịch bản |
| [MIMOTDD_MonteCarlo_Results.xlsx](./MIMOTDD_MonteCarlo_Results.xlsx) | Tệp dữ liệu kết quả mô phỏng và các bảng thống kê |

## 3. Các kịch bản mô phỏng

- **KB1 – Đơn cell:** L = 1, K = 4, f = 1, độ dài pilot τp = 4.
- **KB2 – Đa cell:** L = 4, K = 4, f = 1, độ dài pilot τp = 4.
- **KB3 – Đa cell:** L = 4, K = 4, f = 4, độ dài pilot τp = 16.

Số anten tại mỗi trạm gốc được khảo sát từ M = 10 đến M = 100.

## 4. Các chỉ tiêu đánh giá

- Hiệu suất phổ (SE), đơn vị bit/s/Hz.
- Tỷ số tín hiệu trên nhiễu và can nhiễu (SINR), biểu diễn theo dB trên các đồ thị.
- So sánh ba phương pháp kết hợp tín hiệu MR, ZF và M-MMSE.

## 5. Quy trình thực hiện

1. Chạy chương trình mô phỏng chính bằng Python để tạo dữ liệu kết quả.
2. Sử dụng notebook Python để phân tích và tổng hợp số liệu.
3. Đọc dữ liệu Excel bằng MATLAB Online để tạo đồ thị SE, SINR và đồ thị so sánh ba kịch bản.

Khi chạy MATLAB, đặt file Excel cùng thư mục với script hoặc cập nhật đường dẫn đến file dữ liệu tương ứng.
