module controller_fsm_tb;

    logic       clk;
    logic       rst_n;
    logic       start;
    logic       data_sel;
    logic       data_valid;
    logic       load_en;
    logic [1:0] load_row;
    logic [1:0] load_col;
    logic       data_ready;
    logic [2:0] cycle_count;
    logic       feed_en;
    logic       clear_acc;
    logic       compute_en;
    logic       compute_done;
    logic       output_en;
    logic       out_done;

    initial begin : fsdb_dump
        $fsdbDumpfile("controller_fsm_tb.fsdb");
        $fsdbDumpvars(0, controller_fsm_tb);
        $fsdbDumpMDA();
    end

    controller_fsm dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .data_sel    (data_sel),
        .data_valid  (data_valid),
        .out_done    (out_done),
        .load_en     (load_en),
        .load_row    (load_row),
        .load_col    (load_col),
        .data_ready  (data_ready),
        .cycle_count (cycle_count),
        .feed_en     (feed_en),
        .clear_acc   (clear_acc),
        .compute_en  (compute_en),
        .compute_done(compute_done),
        .output_en   (output_en)
    );

    always #5 clk = ~clk;

    task automatic send_elements(
        input logic select_b
    );
        for (int i = 0; i < 16; i++) begin
            data_sel = select_b;
            @(negedge clk);
            while (!data_ready) begin
                @(negedge clk);
            end
            data_valid = 1'b1;
            #1;
            if (load_en !== 1'b1) begin
                $error("load_en should assert when data_valid is accepted");
            end
            if (load_row !== i[3:2] || load_col !== i[1:0]) begin
                $error(
                    "Load address mismatch at index %0d: got row=%0d col=%0d expected row=%0d col=%0d",
                    i,
                    load_row,
                    load_col,
                    i[3:2],
                    i[1:0]
                );
            end
            @(negedge clk);
            data_valid = 1'b0;
            #1;
            if (load_en !== 1'b0) begin
                $error("load_en should deassert after data_valid drops");
            end
        end
    endtask

    task automatic check_clear_phase;
        while (clear_acc !== 1'b1) begin
            @(posedge clk);
            #1;
        end
        if (feed_en !== 1'b0 || compute_en !== 1'b0 || compute_done !== 1'b0 || output_en !== 1'b0) begin
            $error("CLEAR phase outputs are incorrect");
        end
        @(posedge clk);
        #1;
        if (clear_acc !== 1'b0) begin
            $error("CLEAR phase should last exactly 1 cycle");
        end
    endtask

    task automatic check_compute_phase;
        for (int i = 0; i < 10; i++) begin
            if (compute_en !== 1'b1) begin
                $error("COMPUTE phase should keep compute_en high at cycle %0d", i);
            end
            if (feed_en !== (i < 7)) begin
                $error("feed_en mismatch at compute cycle %0d", i);
            end
            if (compute_done !== (i == 9)) begin
                $error("compute_done mismatch at compute cycle %0d", i);
            end
            if (cycle_count !== i[2:0]) begin
                $error("cycle_count mismatch at compute cycle %0d: got %0d", i, cycle_count);
            end
            if (clear_acc !== 1'b0 || output_en !== 1'b0) begin
                $error("Unexpected outputs during COMPUTE cycle %0d", i);
            end
            if (i < 9) begin
                @(posedge clk);
                #1;
            end
        end
        @(posedge clk);
        #1;
        if (compute_en !== 1'b0) begin
            $error("COMPUTE phase should end after 10 cycles");
        end
    endtask

    task automatic check_output_phase;
        out_done = 1'b0;

        for (int i = 0; i < 3; i++) begin
            if (output_en !== 1'b1) begin
                $error("OUTPUT phase should keep output_en high at cycle %0d", i);
            end
            if (compute_done !== 1'b0) begin
                $error("compute_done should be low during OUTPUT");
            end
            if (clear_acc !== 1'b0 || compute_en !== 1'b0 || feed_en !== 1'b0) begin
                $error("Unexpected control outputs during OUTPUT cycle %0d", i);
            end
            @(posedge clk);
            #1;
        end

        out_done = 1'b1;
        #1;
        if (output_en !== 1'b1) begin
            $error("output_en should remain high while out_done is high in OUTPUT state");
        end

        @(posedge clk);
        #1;
        out_done = 1'b0;
        if (output_en !== 1'b0) begin
            $error("OUTPUT phase should return to IDLE cleanly");
        end
    endtask

    initial begin
        clk          = 1'b0;
        rst_n        = 1'b0;
        start        = 1'b0;
        data_sel     = 1'b0;
        data_valid   = 1'b0;
        out_done     = 1'b0;

        repeat (2) @(posedge clk);
        rst_n = 1'b1;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        send_elements(1'b0);
        send_elements(1'b1);

        check_clear_phase();
        check_compute_phase();
        check_output_phase();

        $display("controller_fsm_tb passed");
        $finish;
    end

endmodule
