module systolic_array_8x8_tb;

    logic        clk;
    logic        rst_n;
    logic        clear_acc;
    logic        compute_en;
    logic [7:0]  a_data [0:7];
    logic [7:0]  b_data [0:7];
    logic signed [19:0] c_data [0:63];

    // Flat storage for $readmemh (row-major, 64 entries each)
    logic [7:0]  A_flat [0:63];
    logic [7:0]  B_flat [0:63];
    logic [31:0] C_exp  [0:63];

    // 2-D views of A and B for skewed feeding
    logic [7:0] a_mat [0:7][0:7];
    logic [7:0] b_mat [0:7][0:7];

    initial begin : fsdb_dump
        $fsdbDumpfile("systolic_array_8x8_tb.fsdb");
        $fsdbDumpvars(0, systolic_array_8x8_tb);
        $fsdbDumpMDA();
    end

    systolic_array_8x8 dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_acc  (clear_acc),
        .compute_en (compute_en),
        .a_data     (a_data),
        .b_data     (b_data),
        .c_data     (c_data)
    );

    always #5 clk = ~clk;

    initial begin
        clk        = 1'b0;
        rst_n      = 1'b0;
        clear_acc  = 1'b0;
        compute_en = 1'b0;

        for (int i = 0; i < 8; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end

        // Load matrices from hex files (one byte/word per line, row-major)
        $readmemh("tile_04_A.hex",        A_flat);
        $readmemh("tile_04_B.hex",        B_flat);
        $readmemh("tile_04_C_expected.hex", C_exp);

        // Reshape flat arrays into 2-D matrices
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) begin
                a_mat[i][j] = A_flat[i*8 + j];
                b_mat[i][j] = B_flat[i*8 + j];
            end

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        clear_acc = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_acc  = 1'b0;
        compute_en = 1'b1;

        // Feed skewed data for 22 cycles (t = 0..21).
        // Valid injection window for 8x8: t = 0..14  (2*(N-1) = 14).
        // Cycles 15..21 drain the last data through a_out/b_out pipeline registers.
        for (int t = 0; t < 22; t++) begin
            for (int r = 0; r < 8; r++) begin
                a_data[r] = ((t >= r) && ((t - r) < 8)) ? a_mat[r][t-r] : 8'd0;
            end
            for (int c = 0; c < 8; c++) begin
                b_data[c] = ((t >= c) && ((t - c) < 8)) ? b_mat[t-c][c] : 8'd0;
            end
            @(posedge clk);
            @(negedge clk);
        end

        compute_en = 1'b0;
        for (int i = 0; i < 8; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end

        // Allow one delta cycle for combinational settling, then check.
        #1;
        begin
            logic signed [31:0] got, exp;
            int pass, fail;
            pass = 0; fail = 0;
            for (int i = 0; i < 64; i++) begin
                // c_data is signed [19:0]; $signed cast widens it to 32 bits correctly.
                got = 32'($signed(c_data[i]));
                exp = $signed(C_exp[i]);
                if (got !== exp) begin
                    $error("FAIL C[%0d][%0d]: got=%0d (0x%05X)  expected=%0d (0x%08X)",
                           i/8, i%8, got, c_data[i], exp, C_exp[i]);
                    fail++;
                end else begin
                    pass++;
                end
            end
            if (fail == 0)
                $display("systolic_array_8x8_tb PASSED (%0d/64 elements correct)", pass);
            else
                $display("systolic_array_8x8_tb FAILED (%0d passed, %0d failed)", pass, fail);
        end

        $finish;
    end

endmodule
