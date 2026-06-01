module systolic_array_4x4_tb;

    logic        clk;
    logic        rst_n;
    logic        clear_acc;
    logic        compute_en;
    logic [7:0]  a_data [0:3];
    logic [7:0]  b_data [0:3];
    logic [17:0] c_data [0:15];

    logic [7:0] a_mat [0:3][0:3];
    logic [7:0] b_mat [0:3][0:3];
    logic [17:0] expected [0:15];

    initial begin : fsdb_dump
        $fsdbDumpfile("systolic_array_4x4_tb.fsdb");
        $fsdbDumpvars(0, systolic_array_4x4_tb);
        $fsdbDumpMDA();
    end

    systolic_array_4x4 dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (clear_acc),
        .compute_en(compute_en),
        .a_data    (a_data),
        .b_data    (b_data),
        .c_data    (c_data)
    );

    always #5 clk = ~clk;

    initial begin
        clk        = 1'b0;
        rst_n      = 1'b0;
        clear_acc  = 1'b0;
        compute_en = 1'b0;

        for (int i = 0; i < 4; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end

        a_mat[0][0] = 8'd1;  a_mat[0][1] = 8'd2;  a_mat[0][2] = 8'd3;  a_mat[0][3] = 8'd4;
        a_mat[1][0] = 8'd5;  a_mat[1][1] = 8'd6;  a_mat[1][2] = 8'd7;  a_mat[1][3] = 8'd8;
        a_mat[2][0] = 8'd9;  a_mat[2][1] = 8'd10; a_mat[2][2] = 8'd11; a_mat[2][3] = 8'd12;
        a_mat[3][0] = 8'd13; a_mat[3][1] = 8'd14; a_mat[3][2] = 8'd15; a_mat[3][3] = 8'd16;

        b_mat[0][0] = 8'd1;  b_mat[0][1] = 8'd0;  b_mat[0][2] = 8'd2;  b_mat[0][3] = 8'd1;
        b_mat[1][0] = 8'd0;  b_mat[1][1] = 8'd1;  b_mat[1][2] = 8'd3;  b_mat[1][3] = 8'd2;
        b_mat[2][0] = 8'd4;  b_mat[2][1] = 8'd1;  b_mat[2][2] = 8'd0;  b_mat[2][3] = 8'd3;
        b_mat[3][0] = 8'd2;  b_mat[3][1] = 8'd5;  b_mat[3][2] = 8'd1;  b_mat[3][3] = 8'd0;

        expected[0]  = 18'd21;  expected[1]  = 18'd25;  expected[2]  = 18'd12;  expected[3]  = 18'd14;
        expected[4]  = 18'd49;  expected[5]  = 18'd53;  expected[6]  = 18'd36;  expected[7]  = 18'd38;
        expected[8]  = 18'd77;  expected[9]  = 18'd81;  expected[10] = 18'd60;  expected[11] = 18'd62;
        expected[12] = 18'd105; expected[13] = 18'd109; expected[14] = 18'd84;  expected[15] = 18'd86;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        clear_acc = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_acc = 1'b0;
        compute_en = 1'b1;

        for (int t = 0; t < 10; t++) begin
            for (int r = 0; r < 4; r++) begin
                a_data[r] = ((t >= r) && ((t - r) < 4)) ? a_mat[r][t-r] : 8'd0;
            end
            for (int c = 0; c < 4; c++) begin
                b_data[c] = ((t >= c) && ((t - c) < 4)) ? b_mat[t-c][c] : 8'd0;
            end
            @(posedge clk);
            @(negedge clk);
        end

        compute_en = 1'b0;
        for (int i = 0; i < 4; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end

        #1;
        for (int i = 0; i < 16; i++) begin
            if (c_data[i] !== expected[i]) begin
                $error("Mismatch at C[%0d]: observed=%0d expected=%0d", i, c_data[i], expected[i]);
            end
        end

        $display("systolic_array_4x4_tb passed");
        $finish;
    end

endmodule
