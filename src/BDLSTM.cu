#include <stdlib.h>
#include <string.h>
#include "BDLSTM.cuh"


double[] SigmoidA(double[] x)
{
    double[] result = new double[x.GetLength(0)];
    for (int i=0; i < x.GetLength(0); i++)
    {
        result[i] = 1.0/(1.0 + Math.Exp(-x[i]));
    }
    return result;
}


double[] matadd1D(double[] m1, double[] m2)
{
    double[] result = new double[m1.GetLength(0)];
    for (int i = 0; i < m1.GetLength(0); i++)
    {
        result[i] = m1[i] + m2[i];
    }
    return result;
}


double[] matmult2(double[]arr1,double[,]arr2)
{
    double temp = 0;
    double[] result = new double[arr2.GetLength(1)];

    for (int i = 0; i < arr2.GetLength(1); i++)
    {
        for (int j = 0; j < arr1.GetLength(0); j++)
        {
            temp += arr1[j] * arr2[j, i];
        }
        result[i] = temp;
        temp = 0;
    }
    return result;
}


double[] mult(double[]arr1,double[]arr2)
{
    double[] result = new double[arr1.GetLength(0)];
    if(arr1.GetLength(0) == arr2.GetLength(0))
    {
        for(int i=0; i < arr1.GetLength(0); i++)
        {
            result[i] = arr1[i] * arr2[i];
        }
    }
    else
    {
        throw new Exception("columns not equal");
    }
    return result;
}


double[] Tanh(double[] arr)
{
    double[] result = new double[arr.GetLength(0)];

    for(int i=0; i<arr.Length; i++)
    {
        result[i] = Math.Tanh(arr[i]);
    }
    return result;
}


__host__ int BDLSTM_cuda(
    double const **const input
  , double const* nnFlat
  , double const* nnLong
  , double const* nnShort
){
    int rc = 0;

    double[] lstm_output = new double[256];
    double[] f_lstm_output = new double[256];
    double[] b_lstm_output = new double[256];
    double[] f_lstm_state = new double[256];
    double[] b_lstm_state = new double[256];

    for (int i = 0; i < lstm_output.Length; i++)
    {
        lstm_output[i] = 0.001;
        f_lstm_output[i] = 0.001;
        b_lstm_output[i] = 0.001;
        f_lstm_state[i] = 0.001;
        b_lstm_state[i] = 0.001;
    }

    nnFlat = nnLong = nnShort = -0.01;

    for (int i = 0; i < _input.Length; i++)
    {
        double[] f_iiput = _input[i];
        double[] b_iiput = _input[_input.Length - i - 1];

        double[] f_input_gate = SigmoidA(matadd1D(matadd1D(matmult2(f_iiput,f_ig_2D),(matmult2(lstm_output,f_ih_2D))),f_bi));
        double[] b_input_gate = SigmoidA(matadd1D(matadd1D(matmult2(b_iiput,b_ig_2D),(matmult2(lstm_output,b_ih_2D))),b_bi));

        double[] f_forget_gate = SigmoidA(matadd1D(matadd1D(matmult2(f_iiput,f_fg_2D),(matmult2(lstm_output,f_fh_2D))),f_bf));
        double[] b_forget_gate = SigmoidA(matadd1D(matadd1D(matmult2(b_iiput,b_fg_2D),(matmult2(lstm_output,b_fh_2D))),b_bf));

        double[] f_output_gate = SigmoidA(matadd1D(matadd1D(matmult2(f_iiput,f_og_2D),(matmult2(lstm_output,f_oh_2D))),f_bo));
        double[] b_output_gate = SigmoidA(matadd1D(matadd1D(matmult2(b_iiput,b_og_2D),(matmult2(lstm_output,b_oh_2D))),f_bo));

        double[] f_memory_cell = Tanh(matadd1D(matadd1D(matmult2(f_iiput,f_mc_2D),(matmult2(lstm_output,f_mh_2D))),f_bm));
        double[] b_memory_cell = Tanh(matadd1D(matadd1D(matmult2(b_iiput,b_mc_2D),(matmult2(lstm_output,b_mh_2D))),b_bm));

        f_lstm_state  = matadd1D((mult(f_lstm_state,f_input_gate)),(mult(f_forget_gate, f_memory_cell)));
        b_lstm_state  = matadd1D((mult(b_lstm_state,b_input_gate)),(mult(b_forget_gate, b_memory_cell)));

        f_lstm_output = mult(f_output_gate,(Tanh(f_lstm_state)));
        b_lstm_output = mult(f_output_gate,(Tanh(b_lstm_state)));

        lstm_output = Tanh(matadd1D(f_lstm_output,b_lstm_output));
    }

    double[] model = matadd1D(matmult2(lstm_output, wo_2D), bol);

    double[] nnModel = Softmax(model);

    nnFlat = nnModel[0];
    nnLong = nnModel[1];
    nnShort= nnModel[2];


    return rc;
}

__global__ int GPU_LSTM()
{

}

