module testbench();
timeunit 10ns;
timeprecision 1ns;

logic reset, Clk;
logic [8:0] xn3, w0, w1, w2;;
logic [8:0] x_hat, err;

always begin : CLOCK_GENERATION
 
 #1 Clk = ~Clk;

end
 
initial begin : CLOCK_INITIALIZATION
	Clk = 0;
end

predictor p(.*);

parameter [3:0] data [4] = {0.19778589 * 16, 0.55186864 * 16, 0.73500254 * 16, -0.69549231 * 16};
//{4, 8, 12, -0.69549231 * 16};

//x_hat should be = -0.69549231
real result;
assign result = x_hat/16/16;

initial begin : TEST_VECTORS

#2 reset = 0;
#2 reset = 1;
#2 reset = 0;

//set weights: [-0.67776111 -0.41394865 -0.17928979]
#2 w0 = -0.67776111*16; w1=-0.41394865*16; w2=-0.17928979*16;

#2 xn3 = data[3]; //xn is the one at the end and xn-1 is the one before that, etc
#2 xn3 = data[2];
#2 xn3 = data[1];
#2 xn3 = data[0];

end

endmodule