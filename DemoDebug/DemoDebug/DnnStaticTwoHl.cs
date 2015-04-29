﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DemoDebug
{
    /* description
     * Both figures illustrate the input-output mechanism for a neural network that has three inputs,
     * a first hidden layer ("A") with four neurons, a second hidden layer ("B") with five neurons and two outputs. 
     * "There are several different meanings for exactly what a deep neural network is, but one is just a neural network 
     * with two (or more) layers of hidden nodes." 3-4-5-2 neural network requires a total of (3 * 4) + 4 + (4 * 5) + 5 + (5 * 2) + 2 = 53 
     * weights and bias values. In the demo, the weights and biases are set to dummy values of 0.01, 0.02, . . . , 0.53. 
     * The three inputs are arbitrarily set to 1.0, 2.0 and 3.0. Behind the scenes, 
     * the neural network uses the hyperbolic tangent activation function when computing the outputs of the two hidden layers,
     * and the softmax activation function when computing the final output values. The two output values are 0.4881 and 0.5119.
     * http://visualstudiomagazine.com/articles/2014/06/01/deep-neural-networks.aspx
     * */
    public class DnnStaticTwoHl
    {
        //using 1 input layer,2 hidden layer and 1 output layer
        private int numInput;
        private int numFirstHidden;
        private int numSecondHidden;
        private int numOutput;

        //value inside input neuron
        private double[] inputs;

        //get weight value for each network
        private double[][] inputToFhWeight;
        private double[][] fhToShWeight;
        private double[][] shToOutWeight;

        private double[] fhBias;
        private double[] shBias;
        private double[] outBias;

        //value after input processed
        private double[] fhOutputs;
        private double[] shOutputs;
        private double[] finalOutputs;

        //set random variable, sometimes usefull for generating random weight
        private static Random rnd;

        //number all weight
        private int numWeights;

        //constructor
        public DnnStaticTwoHl(int numInput,int numFirstHidden, int numSecondHidden, int numOutput)
        {
            //make instance of all variable
            this.numInput = numInput;
            this.numFirstHidden = numFirstHidden;
            this.numSecondHidden = numSecondHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];

            inputToFhWeight = MakeMatrix(numInput, numFirstHidden);
            fhToShWeight = MakeMatrix(numFirstHidden, numSecondHidden);
            shToOutWeight = MakeMatrix(numSecondHidden, numOutput);

            fhBias = new double[numFirstHidden];
            shBias = new double[numSecondHidden];
            outBias = new double[numOutput];

            fhOutputs = new double[numFirstHidden];
            shOutputs = new double[numSecondHidden];
            finalOutputs = new double[numOutput];

            rnd = new Random(0);
            this.numWeights = (numInput * numFirstHidden) + numFirstHidden + (numFirstHidden * numSecondHidden) + numSecondHidden + (numSecondHidden * numOutput) + numOutput;

            InitWeight();
        }

        //making metrices of network
        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        //helper for random range
        private double RandomRange(double min, double max)
        {
            return (min - max) * rnd.NextDouble() + min; 
        }

        //initialize all wight value;
        private void InitWeight()
        {
            double[] weight = new double[numWeights];
            double min = -0.01;
            double max = 0.01;
            for (int i = 0; i < weight.Length; ++i)
                weight[i] = RandomRange(min, max);

            this.SetWeights(weight);
        }

        //store all weight for each neuron
        public void SetWeights(double[] weights)
        {
           if (weights.Length != numWeights)
                throw new Exception("Bad weights length");

            int k = 0;

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numFirstHidden; ++j)
                    inputToFhWeight[i][j] = weights[k++];

            for (int i = 0; i < numFirstHidden; ++i)
                fhBias[i] = weights[k++];

            for (int i = 0; i < numFirstHidden; ++i)
                for (int j = 0; j < numSecondHidden; ++j)
                    fhToShWeight[i][j] = weights[k++];

            for (int i = 0; i < numSecondHidden; ++i)
                shBias[i] = weights[k++];

            for (int i = 0; i < numSecondHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    shToOutWeight[i][j] = weights[k++];

            for (int i = 0; i < numOutput; ++i)
                outBias[i] = weights[k++];
        }
        
        //computeoutput
        public double[] ComputeOutputs(double[] xValues)
        {
            double[] firstSums = new double[numFirstHidden]; // first hidden nodes sums scratch array
            double[] secondSums = new double[numSecondHidden]; // second hidden nodes sums scratch array
            double[] outSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numFirstHidden; ++j)  // compute sum of (ia) weights * inputs
                for (int i = 0; i < numInput; ++i)
                    firstSums[j] += this.inputs[i] * this.inputToFhWeight[i][j]; // note +=

            for (int i = 0; i < numFirstHidden; ++i)  // add biases to a sums
                firstSums[i] += this.fhBias[i];

            Console.WriteLine("\nInternal aSums:");
            ShowVector(firstSums, firstSums.Length, 4, true);

            for (int i = 0; i < numFirstHidden; ++i)   // apply activation
                this.fhOutputs[i] = HyperTanFunction(firstSums[i]); // hard-coded

            Console.WriteLine("\nInternal first hidden Outputs:");
            ShowVector(fhOutputs, fhOutputs.Length, 4, true);

            for (int j = 0; j < numSecondHidden; ++j)  // compute sum of (ab) weights * a outputs = local inputs
                for (int i = 0; i < numFirstHidden; ++i)
                    secondSums[j] += fhOutputs[i] * this.fhToShWeight[i][j]; // note +=

            for (int i = 0; i < numSecondHidden; ++i)  // add biases to b sums
                secondSums[i] += this.shBias[i];

            Console.WriteLine("\nInternal bSums:");
            ShowVector(secondSums, secondSums.Length, 4, true);

            for (int i = 0; i < numSecondHidden; ++i)   // apply activation
                this.shOutputs[i] = HyperTanFunction(secondSums[i]); // hard-coded

            Console.WriteLine("\nInternal bOutputs:");
            ShowVector(shOutputs, shOutputs.Length, 4, true);

            for (int j = 0; j < numOutput; ++j)   // compute sum of (bo) weights * b outputs = local inputs
                for (int i = 0; i < numSecondHidden; ++i)
                    outSums[j] += shOutputs[i] * shToOutWeight[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                outSums[i] += outBias[i];

            Console.WriteLine("\nInternal oSums:");
            ShowVector(outSums, outSums.Length, 4, true);

            double[] softOut = Softmax(outSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, finalOutputs, softOut.Length);

            double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
            Array.Copy(this.finalOutputs, retResult, retResult.Length);
            return retResult;

        }

        //Hyperbolic Tangent Activation Function
        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // hardcoded approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        //Softmax activation function
        private static double[] Softmax(double[] oSums)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }
        static public void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }
    }
}
