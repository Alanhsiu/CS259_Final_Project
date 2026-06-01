module buffer_input (
    input  logic       clk,
    input  logic       rst_n,

    input  logic       load_en,
    input  logic       load_sel,
    input  logic [2:0] load_row,
    input  logic [2:0] load_col,
    input  logic [7:0] load_data,

    input  logic [3:0] cycle_count,

    input  logic       feed_en,
    output logic [7:0] a_data [0:7],
    output logic [7:0] b_data [0:7]
);

    logic [7:0] A_buf [0:7][0:7];
    logic [7:0] B_buf [0:7][0:7];

    integer i, j;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 8; i++) begin
                for (j = 0; j < 8; j++) begin
                    A_buf[i][j] <= 8'd0;
                    B_buf[i][j] <= 8'd0;
                end
            end
        end else if (load_en) begin
            if (!load_sel)
                A_buf[load_row][load_col] <= load_data;
            else
                B_buf[load_row][load_col] <= load_data;
        end
    end

    always_comb begin
        for (int r = 0; r < 8; r++) begin
            a_data[r] = 8'd0;
        end
        for (int c = 0; c < 8; c++) begin
            b_data[c] = 8'd0;
        end

        if (feed_en) begin
            for (int r = 0; r < 8; r++) begin
                if ((cycle_count >= r) && ((cycle_count - r) < 8)) begin
                    a_data[r] = A_buf[r][cycle_count - r];
                end
            end

            for (int c = 0; c < 8; c++) begin
                if ((cycle_count >= c) && ((cycle_count - c) < 8)) begin
                    b_data[c] = B_buf[cycle_count - c][c];
                end
            end
        end
    end

endmodule
