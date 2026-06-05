// attention_tb.sv
// Two-test suite for attention_top (N=8, NTILES=8, d_k=64, SCALE_BITS=3).
//
// ── Test 1  (Sanity / closed-form) ───────────────────────────────────────
//   Q = K^T = all-1s,  V[r][c] = r   (same tile loaded all 8 times)
//   accum per element = 8 tiles × (8 × 1×1) = 64
//   score_rf = 64 >>> 3 = 8  → uniform softmax → attn_q07 = 16
//   out[*][*] = Σ_k 16·k = 16·28 = 448
//
// ── Test 2  (Real data, full d_k=64) ────────────────────────────────────
//   Q  ← real_q_tile.hex  (8×64 row-major INT8)
//   K  ← real_k_tile.hex  (8×64 row-major INT8)
//   Hardware runs 8 KT tiles; accum_rf = full Q·K^T (d_k=64)
//   score_rf = accum_rf >>> 3
//   Verification: dut.score_rf[r][c] === c_exp[r*8+c] >>> 3  (exact)

`timescale 1ns/1ps

module attention_tb;

    // ---------------------------------------------------------------
    // Parameters
    // ---------------------------------------------------------------
    localparam int N          = 8;
    localparam int NTILES     = 8;
    localparam int SCALE_BITS = 3;
    localparam int WIN        = 20;
    localparam int WOUT       = 16;
    localparam int CLK_PERIOD = 10;
    localparam int TIMEOUT_CY = 8000;

    // ---------------------------------------------------------------
    // DUT wires
    // ---------------------------------------------------------------
    logic        clk, rst_n;
    logic [1:0]  load_sel;
    logic        load_en;
    logic [2:0]  load_row, load_col;
    logic [7:0]  load_data;
    logic        start;
    logic        out_ready;
    logic signed [WIN-1:0] out_data;
    logic        out_valid, tile_rdy, done;

    // ---------------------------------------------------------------
    // DUT
    // ---------------------------------------------------------------
    attention_top #(
        .N         (N),
        .WIN       (WIN),
        .WOUT      (WOUT),
        .NTILES    (NTILES),
        .SCALE_BITS(SCALE_BITS)
    ) dut (
        .clk      (clk),       .rst_n    (rst_n),
        .load_sel (load_sel),  .load_en  (load_en),
        .load_row (load_row),  .load_col (load_col),
        .load_data(load_data), .start    (start),
        .out_ready(out_ready), .out_data (out_data),
        .out_valid(out_valid), .tile_rdy (tile_rdy),
        .done     (done)
    );

    // ---------------------------------------------------------------
    // Clock
    // ---------------------------------------------------------------
    initial clk = 1'b0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---------------------------------------------------------------
    // Output capture
    // ---------------------------------------------------------------
    logic signed [WIN-1:0] result_buf [0:N*N-1];
    int out_cnt;

    always_ff @(posedge clk) begin
        if (!rst_n)
            out_cnt <= 0;
        else if (out_valid && out_ready) begin
            result_buf[out_cnt] <= out_data;
            out_cnt             <= out_cnt + 1;
        end
    end

    // ---------------------------------------------------------------
    // Common tasks
    // ---------------------------------------------------------------
    task automatic load_matrix(
        input [1:0]       sel,
        input logic [7:0] mat [0:N-1][0:N-1]
    );
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) begin
                @(negedge clk);
                load_sel  = sel;
                load_en   = 1'b1;
                load_row  = 3'(r);
                load_col  = 3'(c);
                load_data = mat[r][c];
            end
        @(negedge clk);
        load_en = 1'b0;
    endtask

    task automatic do_start();
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
    endtask

    task automatic do_reset();
        @(negedge clk); rst_n = 1'b0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        repeat(2) @(posedge clk);
    endtask

    task automatic wait_done_or_timeout(input int test_id);
        fork
            begin : w_done
                @(posedge done);
                disable w_tout;
            end
            begin : w_tout
                repeat(TIMEOUT_CY) @(posedge clk);
                $display("[TIMEOUT] Test %0d exceeded %0d cycles", test_id, TIMEOUT_CY);
                $finish;
            end
        join
        repeat(3) @(posedge clk);
    endtask

    // tile_rdy fires after tiles 0..NTILES-2; host reloads KT and asserts start again
    task automatic wait_tile_rdy_or_timeout(input int test_id);
        fork
            begin : w_tile
                @(posedge tile_rdy);
                disable w_tout_tile;
            end
            begin : w_tout_tile
                repeat(TIMEOUT_CY) @(posedge clk);
                $display("[TIMEOUT] Tile in test %0d exceeded %0d cycles", test_id, TIMEOUT_CY);
                $finish;
            end
        join
        repeat(2) @(posedge clk);  // let FSM return to IDLE
    endtask

    // ---------------------------------------------------------------
    // TEST 1 — Uniform attention, closed-form (8 identical tiles)
    // ---------------------------------------------------------------
    task automatic run_test1();
        logic [7:0] Q1  [0:N-1][0:N-1];
        logic [7:0] KT1 [0:N-1][0:N-1];
        logic [7:0] V1  [0:N-1][0:N-1];
        int errs;

        $display("\n====== TEST 1: Uniform Attention (8 identical tiles, closed-form) ======");
        $display("  Q=1  KT=1  V[r][c]=r");
        $display("  accum = 8 tiles × 8 = 64  →  score_rf = 64>>>3 = 8 (uniform)");
        $display("  attn_q07 = 16  →  out = 16×28 = 448");

        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) begin
                Q1 [r][c] = 8'h01;
                KT1[r][c] = 8'h01;
                V1 [r][c] = 8'(r);
            end

        do_reset();
        load_matrix(2'b10, V1);  // V loaded once; stays for all tiles

        for (int t = 0; t < NTILES; t++) begin
            // Q and KT are the same every tile for this test
            load_matrix(2'b00, Q1);
            load_matrix(2'b01, KT1);
            do_start();
            if (t < NTILES - 1)
                wait_tile_rdy_or_timeout(1);
            else
                wait_done_or_timeout(1);
        end

        errs = 0;
        $display("  Raw outputs (expected 448):");
        for (int r = 0; r < N; r++) begin
            $write("    row%0d:", r);
            for (int c = 0; c < N; c++) begin
                $write(" %6d", result_buf[r*N+c]);
                if (result_buf[r*N+c] !== 20'sd448) errs++;
            end
            $write("\n");
        end

        if (errs == 0) $display("  [PASS] All 64 outputs = 448");
        else           $display("  [FAIL] %0d mismatches", errs);
    endtask

    // ---------------------------------------------------------------
    // TEST 2 — Real data, full d_k=64, tiled HW vs c_expected
    // ---------------------------------------------------------------
    task automatic run_test2();
        // Hex file contents
        logic [7:0]          Q_raw [0:511];          // Q[8×64] row-major
        logic [7:0]          K_raw [0:511];          // K[8×64] row-major
        logic signed [31:0]  c_exp [0:N*N-1];        // full Q·K^T golden (d_k=64)

        // Per-tile slices (loaded into hardware buffers)
        logic [7:0]          Q_sl  [0:N-1][0:N-1];   // Q columns t*8 .. t*8+7
        logic [7:0]          KT_sl [0:N-1][0:N-1];   // KT tile t
        logic [7:0]          V_eye [0:N-1][0:N-1];   // identity V

        logic signed [WIN-1:0] exp_score;
        int errs;

        $display("\n====== TEST 2: Real Data d_k=64 (8 tiles), score_rf vs c_expected >>> 3 ======");

        // ---- Load hex files ----
        $readmemh("real_q_tile.hex", Q_raw);
        $readmemh("real_k_tile.hex", K_raw);
        $readmemh("real_c_expected.hex", c_exp);

        // ---- Identity V ----
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
                V_eye[r][c] = (r == c) ? 8'd1 : 8'd0;

        // ---- Tiled hardware run ----
        // Reset once.  Load V once.
        // For each tile: load Q slice + KT slice, assert start.
        // Wait tile_rdy after tiles 0..6; wait done after tile 7.
        do_reset();
        load_matrix(2'b10, V_eye);

        for (int t = 0; t < NTILES; t++) begin
            // Q slice: Q[r][t*8 .. t*8+7]  stored row-major in Q_raw at offset r*64
            for (int r = 0; r < N; r++)
                for (int k = 0; k < N; k++)
                    Q_sl[r][k] = Q_raw[r*64 + t*8 + k];

            // KT slice: KT[k][c] = K[c][t*8+k]  (transpose of K slice)
            for (int k = 0; k < N; k++)
                for (int c = 0; c < N; c++)
                    KT_sl[k][c] = K_raw[c*64 + t*8 + k];

            load_matrix(2'b00, Q_sl);
            load_matrix(2'b01, KT_sl);
            do_start();

            if (t < NTILES - 1)
                wait_tile_rdy_or_timeout(2);
            else
                wait_done_or_timeout(2);
        end

        $display("  All 8 tiles complete.");

        // ---- Verify score_rf === c_exp >>> SCALE_BITS ----
        // accum_rf = Σ_t c_data_t = full Q·K^T (exact, 32-bit accumulation)
        // score_rf = accum_rf >>> SCALE_BITS
        // c_exp    = full Q·K^T from external golden
        // → expected: score_rf[r][c] === c_exp[r*8+c] >>> SCALE_BITS  (exact match)
        $display("  score_rf (hw) vs c_expected >>> %0d:", SCALE_BITS);
        errs = 0;
        for (int r = 0; r < N; r++) begin
            $write("    row%0d:", r);
            for (int c = 0; c < N; c++) begin
                exp_score = WIN'($signed(c_exp[r*N+c]) >>> SCALE_BITS);
                if (dut.score_rf[r][c] !== exp_score) begin
                    $write(" [hw=%0d exp=%0d!]", dut.score_rf[r][c], exp_score);
                    errs++;
                end else begin
                    $write(" %7d✓", dut.score_rf[r][c]);
                end
            end
            $write("\n");
        end

        if (errs == 0)
            $display("  [PASS] All 64 score_rf match c_expected >>> %0d", SCALE_BITS);
        else
            $display("  [FAIL] %0d score_rf mismatches", errs);

        // ---- Final outputs (attn·V, V=I → ≈ scaled attention weights) ----
        $display("  Final outputs (attn·V, V=I):");
        for (int r = 0; r < N; r++) begin
            $write("    row%0d:", r);
            for (int c = 0; c < N; c++)
                $write(" %6d", result_buf[r*N+c]);
            $write("\n");
        end
    endtask

    // ---------------------------------------------------------------
    // Watchdog  (NTILES × 8 runs per test, generous margin)
    // ---------------------------------------------------------------
    initial begin
        #(CLK_PERIOD * TIMEOUT_CY * NTILES * 3);
        $display("[WATCHDOG] Absolute timeout");
        $finish;
    end

    // ---------------------------------------------------------------
    // Waveform
    // ---------------------------------------------------------------
    initial begin
        $dumpfile("attention_tb.vcd");
        $dumpvars(0, attention_tb);
    end

    // ---------------------------------------------------------------
    // Main
    // ---------------------------------------------------------------
    initial begin
        load_en = 0; load_sel = 0;
        load_row = 0; load_col = 0; load_data = 0;
        start = 0; out_ready = 1; rst_n = 1;

        run_test1();
        run_test2();

        $display("\n[TB] All tests complete.");
        $finish;
    end

endmodule
