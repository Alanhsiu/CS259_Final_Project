// softmax_unit.sv
// Hardware softmax for a row of N signed integers from the systolic array.
//
// exp() uses 2^x decomposition + 8-bit LUT:
//   exp(-z) = EXP_ROM[w_frac] >> w_int
//   w = z*log2(e),  w_int=floor(w),  w_frac=top-8-bits of frac(w)
//   EXP_ROM[k] = round(2^(-k/256)*32768)  Q1.15, 256 entries
//
// Normalization uses a RIGHT-SHIFT restoring divider:
//   Fixed dividend  = 2^(WEXP+EXTRA) = 2^23  (constant)
//   Scaled divisor  starts at sum_exp<<(WRECIP-1), right-shifts one bit per cycle
//   Each cycle: if dividend_rem >= scaled_div → subtract, Q-bit=1; else Q-bit=0
//   Quotient Q = 2^23 / sum_exp (WRECIP = 9 cycles)
//   out[i] = (exp_val[i] * Q) >> EXTRA  →  Q1.15 probability
//
// Output: Q1.15  (32768 = 1.0; probabilities always sum to 32768)
//
// Latency:  4 + WRECIP + N  cycles  (WRECIP = EXTRA+1 = 9 by default)
//   N=4: 17 cy   N=8: 21 cy   N=16: 29 cy

module softmax_unit #(
    parameter  N    = 4,   // row size: 4, 8, or 16
    parameter  WIN  = 18,  // signed input width (pe.sv acc_out)
    parameter  WOUT = 16   // output width (Q1.15; range [0, 32768])
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            start,
    input  logic [WIN-1:0]  data_in  [N],
    output logic            done,
    output logic [WOUT-1:0] data_out [N]
);

    // ---------------------------------------------------------------
    // Parameters
    // ---------------------------------------------------------------
    localparam int WEXP   = 15;
    localparam int CLOG2N = $clog2(N);
    localparam int WSUM   = WEXP + CLOG2N + 1;  // sum width, no overflow for N<=16
    localparam int EXTRA  = 8;                   // reciprocal guard bits
    localparam int WRECIP = EXTRA + 1;           // quotient bits  (max Q = 2^EXTRA = 256)
    localparam int WDIV_P = WEXP + EXTRA + 1;   // running-dividend register width = 24
    localparam int WDIV_D = WSUM + WRECIP - 1;  // shifted-divisor register width
    localparam int WMUL   = WDIV_P;             // product width (24 bits)

    // Fixed dividend: 2^(WEXP+EXTRA) = 2^23
    localparam logic [WDIV_P-1:0] DIVIDEND = WDIV_P'(1) << (WEXP + EXTRA);

    // log2(e) * 2^15 = 47274  (Q0.15)
    localparam logic [15:0] LOG2E_Q15 = 16'd47274;

    // ---------------------------------------------------------------
    // EXP LUT: EXP_ROM[k] = round(2^(-k/256)*32768) for k=0..255
    // ---------------------------------------------------------------
    localparam logic [15:0] EXP_ROM [256] = '{
        16'd32768, 16'd32679, 16'd32591, 16'd32503, 16'd32415, 16'd32327, 16'd32240, 16'd32153,
        16'd32066, 16'd31979, 16'd31893, 16'd31806, 16'd31720, 16'd31635, 16'd31549, 16'd31464,
        16'd31379, 16'd31294, 16'd31209, 16'd31125, 16'd31041, 16'd30957, 16'd30873, 16'd30790,
        16'd30706, 16'd30623, 16'd30541, 16'd30458, 16'd30376, 16'd30293, 16'd30212, 16'd30130,
        16'd30048, 16'd29967, 16'd29886, 16'd29805, 16'd29725, 16'd29644, 16'd29564, 16'd29484,
        16'd29405, 16'd29325, 16'd29246, 16'd29167, 16'd29088, 16'd29009, 16'd28931, 16'd28852,
        16'd28774, 16'd28697, 16'd28619, 16'd28542, 16'd28464, 16'd28388, 16'd28311, 16'd28234,
        16'd28158, 16'd28082, 16'd28006, 16'd27930, 16'd27855, 16'd27779, 16'd27704, 16'd27629,
        16'd27554, 16'd27480, 16'd27406, 16'd27332, 16'd27258, 16'd27184, 16'd27110, 16'd27037,
        16'd26964, 16'd26891, 16'd26818, 16'd26746, 16'd26674, 16'd26601, 16'd26530, 16'd26458,
        16'd26386, 16'd26315, 16'd26244, 16'd26173, 16'd26102, 16'd26031, 16'd25961, 16'd25891,
        16'd25821, 16'd25751, 16'd25681, 16'd25612, 16'd25543, 16'd25474, 16'd25405, 16'd25336,
        16'd25268, 16'd25199, 16'd25131, 16'd25063, 16'd24995, 16'd24928, 16'd24860, 16'd24793,
        16'd24726, 16'd24659, 16'd24593, 16'd24526, 16'd24460, 16'd24394, 16'd24328, 16'd24262,
        16'd24196, 16'd24131, 16'd24066, 16'd24001, 16'd23936, 16'd23871, 16'd23806, 16'd23742,
        16'd23678, 16'd23614, 16'd23550, 16'd23486, 16'd23423, 16'd23359, 16'd23296, 16'd23233,
        16'd23170, 16'd23108, 16'd23045, 16'd22983, 16'd22921, 16'd22859, 16'd22797, 16'd22735,
        16'd22674, 16'd22613, 16'd22552, 16'd22491, 16'd22430, 16'd22369, 16'd22309, 16'd22248,
        16'd22188, 16'd22128, 16'd22068, 16'd22009, 16'd21949, 16'd21890, 16'd21831, 16'd21772,
        16'd21713, 16'd21654, 16'd21595, 16'd21537, 16'd21479, 16'd21421, 16'd21363, 16'd21305,
        16'd21247, 16'd21190, 16'd21133, 16'd21076, 16'd21019, 16'd20962, 16'd20905, 16'd20849,
        16'd20792, 16'd20736, 16'd20680, 16'd20624, 16'd20568, 16'd20513, 16'd20457, 16'd20402,
        16'd20347, 16'd20292, 16'd20237, 16'd20182, 16'd20127, 16'd20073, 16'd20019, 16'd19965,
        16'd19911, 16'd19857, 16'd19803, 16'd19750, 16'd19696, 16'd19643, 16'd19590, 16'd19537,
        16'd19484, 16'd19431, 16'd19379, 16'd19326, 16'd19274, 16'd19222, 16'd19170, 16'd19118,
        16'd19066, 16'd19015, 16'd18963, 16'd18912, 16'd18861, 16'd18810, 16'd18759, 16'd18708,
        16'd18658, 16'd18607, 16'd18557, 16'd18507, 16'd18457, 16'd18407, 16'd18357, 16'd18308,
        16'd18258, 16'd18209, 16'd18160, 16'd18110, 16'd18061, 16'd18013, 16'd17964, 16'd17915,
        16'd17867, 16'd17819, 16'd17770, 16'd17722, 16'd17674, 16'd17627, 16'd17579, 16'd17531,
        16'd17484, 16'd17437, 16'd17390, 16'd17343, 16'd17296, 16'd17249, 16'd17202, 16'd17156,
        16'd17109, 16'd17063, 16'd17017, 16'd16971, 16'd16925, 16'd16879, 16'd16834, 16'd16788,
        16'd16743, 16'd16697, 16'd16652, 16'd16607, 16'd16562, 16'd16518, 16'd16473, 16'd16428
    };

    // ---------------------------------------------------------------
    // exp_approx: compute exp(x_in) for signed integer x_in
    // Returns Q1.15 (16-bit); max = 32768 when x_in = 0.
    // ---------------------------------------------------------------
    function automatic logic [15:0] exp_approx(
        input logic signed [WIN-1:0] x_in
    );
        logic [WIN-1:0]  neg_x;
        logic [WIN+15:0] y_fp;
        logic [WIN-1:0]  w_int;
        logic [7:0]      w_frac;
        logic [15:0]     mantissa;

        neg_x    = ($signed(x_in) <= 0) ? WIN'(unsigned'(-$signed(x_in))) : '0;
        y_fp     = {{16{1'b0}}, neg_x} * {{WIN{1'b0}}, LOG2E_Q15};
        w_int    = y_fp[WIN+14 : 15];
        w_frac   = y_fp[14    :  7];
        mantissa = EXP_ROM[w_frac];

        exp_approx = (w_int >= 16) ? 16'd0 : (mantissa >> w_int);
    endfunction

    // ---------------------------------------------------------------
    // Combinational max tree
    // ---------------------------------------------------------------
    logic signed [WIN-1:0] score [N];
    logic signed [WIN-1:0] max_comb;
    always_comb begin
        max_comb = $signed(score[0]);
        for (int i = 1; i < N; i++)
            if ($signed(score[i]) > $signed(max_comb))
                max_comb = $signed(score[i]);
    end

    // ---------------------------------------------------------------
    // Combinational sum tree
    // ---------------------------------------------------------------
    logic [15:0]     exp_val  [N];
    logic [WSUM-1:0] sum_comb;
    always_comb begin
        sum_comb = '0;
        for (int i = 0; i < N; i++)
            sum_comb = sum_comb + WSUM'(exp_val[i]);
    end

    // ---------------------------------------------------------------
    // Right-shift restoring divider combinational wires
    // div_rem: running remainder (WDIV_P bits, starts at DIVIDEND = 2^23)
    // div_D:   scaled divisor   (WDIV_D bits, starts at sum_exp<<(WRECIP-1))
    // ---------------------------------------------------------------
    logic [WDIV_P-1:0] div_rem;
    logic [WDIV_D-1:0] div_D;
    logic [WRECIP-1:0] div_quot;
    logic [3:0]        div_cnt;

    // Compare: zero-extend div_rem to WDIV_D bits before comparing
    logic div_ge;
    assign div_ge = ({{(WDIV_D-WDIV_P){1'b0}}, div_rem} >= div_D);

    // Subtraction (safe to truncate div_D because when div_ge=1, div_D < 2^WDIV_P)
    logic [WDIV_P-1:0] div_sub;
    assign div_sub = div_rem - div_D[WDIV_P-1:0];

    // ---------------------------------------------------------------
    // Multiply step (combinational, indexed by norm_cnt)
    // ---------------------------------------------------------------
    logic [CLOG2N-1:0] norm_cnt;
    logic [WMUL-1:0]   mul_prod;
    assign mul_prod = WMUL'(exp_val[norm_cnt]) * WMUL'(div_quot);

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    typedef enum logic [2:0] {
        IDLE, LATCH, MAX_REG, EXP_REG, SUM_REG,
        DIV_RECIP, MUL_NORM, DONE_ST
    } state_t;

    state_t state, next_state;
    logic signed [WIN-1:0] max_val;
    logic [WSUM-1:0]       sum_exp;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next_state;
    end

    always_comb begin
        next_state = state;
        case (state)
            IDLE:      if (start)               next_state = LATCH;
            LATCH:                               next_state = MAX_REG;
            MAX_REG:                             next_state = EXP_REG;
            EXP_REG:                             next_state = SUM_REG;
            SUM_REG:                             next_state = DIV_RECIP;
            DIV_RECIP: if (div_cnt == WRECIP-1) next_state = MUL_NORM;
            MUL_NORM:  if (norm_cnt == N-1)      next_state = DONE_ST;
            DONE_ST:                             next_state = IDLE;
            default:                             next_state = IDLE;
        endcase
    end

    // ---------------------------------------------------------------
    // Datapath
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++) begin
                score[i]    <= '0;
                exp_val[i]  <= '0;
                data_out[i] <= '0;
            end
            max_val  <= '0;
            sum_exp  <= '0;
            div_rem  <= '0;
            div_D    <= '0;
            div_quot <= '0;
            div_cnt  <= '0;
            norm_cnt <= '0;
            done     <= 1'b0;
        end
        else begin
            done <= (state == DONE_ST);

            case (state)
                LATCH: begin
                    for (int i = 0; i < N; i++)
                        score[i] <= data_in[i];
                end

                MAX_REG: begin
                    max_val <= max_comb;
                end

                EXP_REG: begin
                    for (int i = 0; i < N; i++)
                        exp_val[i] <= exp_approx($signed(score[i]) - $signed(max_val));
                end

                SUM_REG: begin
                    sum_exp  <= sum_comb;
                    div_rem  <= DIVIDEND;
                    // Initialize scaled divisor to sum_exp << (WRECIP-1)
                    div_D    <= {sum_comb, {(WRECIP-1){1'b0}}};
                    div_quot <= '0;
                    div_cnt  <= '0;
                end

                // Right-shift restoring divider (one quotient bit per cycle, MSB first).
                // Invariant: div_rem < 2^WDIV_P throughout.
                DIV_RECIP: begin
                    if (div_ge) begin
                        div_rem  <= div_sub;
                        div_quot <= WRECIP'((div_quot << 1) | 1'b1);
                    end else begin
                        div_quot <= WRECIP'(div_quot << 1);
                    end
                    div_D   <= div_D >> 1;   // halve scaled divisor each cycle
                    div_cnt <= div_cnt + 4'd1;
                end

                // Sequential normalization: out[i] = (exp[i] * Q) >> EXTRA
                MUL_NORM: begin
                    data_out[norm_cnt] <= WOUT'(mul_prod >> EXTRA);
                    norm_cnt           <= CLOG2N'(norm_cnt + 1'b1);
                end

                DONE_ST: begin
                    norm_cnt <= '0;
                end

                default: ;
            endcase
        end
    end

endmodule
