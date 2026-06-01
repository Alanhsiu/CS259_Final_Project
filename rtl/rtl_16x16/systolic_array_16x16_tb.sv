module systolic_array_16x16_tb;

    // ------------------------------------------------------------------ //
    //  DUT signals                                                         //
    // ------------------------------------------------------------------ //
    logic        clk;
    logic        rst_n;
    logic        clear_acc;
    logic        compute_en;
    logic [7:0]  a_data [0:15];
    logic [7:0]  b_data [0:15];
    logic signed [19:0] c_data [0:255];

    // ------------------------------------------------------------------ //
    //  Storage for one test pattern (shared, written before each tile)     //
    // ------------------------------------------------------------------ //
    logic [7:0]  A_flat [0:255];
    logic [7:0]  B_flat [0:255];
    logic [31:0] C_exp  [0:255];

    logic [7:0]  a_mat  [0:15][0:15];
    logic [7:0]  b_mat  [0:15][0:15];

    // ------------------------------------------------------------------ //
    //  DUT instantiation                                                   //
    // ------------------------------------------------------------------ //
    systolic_array_16x16 dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_acc  (clear_acc),
        .compute_en (compute_en),
        .a_data     (a_data),
        .b_data     (b_data),
        .c_data     (c_data)
    );

    always #5 clk = ~clk;

    // ------------------------------------------------------------------ //
    //  Task: run one tile and return per-element fail count               //
    // ------------------------------------------------------------------ //
    task automatic run_tile(
        input  int tile_id,
        output int n_fail
    );
        string fname_a, fname_b, fname_c;
        logic signed [31:0] got, exp;
        int n_pass;

        $sformat(fname_a, "test_pattern/tile_%02d_A.hex",         tile_id);
        $sformat(fname_b, "test_pattern/tile_%02d_B.hex",         tile_id);
        $sformat(fname_c, "test_pattern/tile_%02d_C_expected.hex", tile_id);

        $readmemh(fname_a, A_flat);
        $readmemh(fname_b, B_flat);
        $readmemh(fname_c, C_exp);

        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 16; j++) begin
                a_mat[i][j] = A_flat[i*16 + j];
                b_mat[i][j] = B_flat[i*16 + j];
            end

        // --- reset DUT between tiles ---
        rst_n      = 1'b0;
        clear_acc  = 1'b0;
        compute_en = 1'b0;
        for (int i = 0; i < 16; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end
        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        // --- clear accumulators ---
        @(negedge clk);
        clear_acc = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_acc  = 1'b0;
        compute_en = 1'b1;

        // --- skewed feed: 46 cycles (t = 0..45) ---
        // valid injection window t = 0..30  (= 2*(N-1), N=16)
        // drain window           t = 31..45 (= N-1)
        for (int t = 0; t < 46; t++) begin
            for (int r = 0; r < 16; r++)
                a_data[r] = ((t >= r) && ((t - r) < 16)) ? a_mat[r][t-r] : 8'd0;
            for (int c = 0; c < 16; c++)
                b_data[c] = ((t >= c) && ((t - c) < 16)) ? b_mat[t-c][c] : 8'd0;
            @(posedge clk);
            @(negedge clk);
        end

        compute_en = 1'b0;
        for (int i = 0; i < 16; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end

        // --- check all 256 outputs ---
        #1;
        n_pass = 0; n_fail = 0;
        for (int i = 0; i < 256; i++) begin
            got = 32'($signed(c_data[i]));
            exp = $signed(C_exp[i]);
            if (got !== exp) begin
                $error("tile_%02d FAIL C[%02d][%02d]: got=%0d  expected=%0d",
                       tile_id, i/16, i%16, got, exp);
                n_fail++;
            end else begin
                n_pass++;
            end
        end

        if (n_fail == 0)
            $display("  tile_%02d  PASS  (256/256 correct)", tile_id);
        else
            $display("  tile_%02d  FAIL  (%0d/256 correct, %0d wrong)",
                     tile_id, n_pass, n_fail);
    endtask

    // ------------------------------------------------------------------ //
    //  Main: run all 10 tiles and print aggregate summary                  //
    // ------------------------------------------------------------------ //
    initial begin : fsdb_dump
        $fsdbDumpfile("systolic_array_16x16_tb.fsdb");
        $fsdbDumpvars(0, systolic_array_16x16_tb);
        $fsdbDumpMDA();
    end

    initial begin
        automatic int tile_fail  = 0;
        automatic int total_fail = 0;
        automatic int tiles_pass = 0;
        automatic int tiles_fail = 0;

        clk        = 1'b0;
        rst_n      = 1'b0;
        clear_acc  = 1'b0;
        compute_en = 1'b0;
        for (int i = 0; i < 16; i++) begin
            a_data[i] = 8'd0;
            b_data[i] = 8'd0;
        end

        $display("=======================================================");
        $display("  systolic_array_16x16 — 10 tile regression            ");
        $display("=======================================================");

        for (int tile = 0; tile <= 9; tile++) begin
            run_tile(tile, tile_fail);
            total_fail += tile_fail;
            if (tile_fail == 0) tiles_pass++;
            else                 tiles_fail++;
        end

        $display("=======================================================");
        if (total_fail == 0)
            $display("  RESULT: ALL PASS  (%0d/10 tiles, 2560/2560 elements)",
                     tiles_pass);
        else
            $display("  RESULT: FAIL  (%0d tiles passed, %0d tiles failed, %0d total element errors)",
                     tiles_pass, tiles_fail, total_fail);
        $display("=======================================================");

        $finish;
    end

endmodule
