using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DemoDebug
{
    class Program
    {
        static void Main(string[] args)
        {
            //testDNN();
            testDNNTraining();
        }

        public static void testDNN()
        {
            Console.WriteLine("\nBegin Deep Neural Network input-output demo");

            Console.WriteLine("\nCreating a 3-4-5-2 neural network");
            int numInput = 3;
            int numHiddenA = 4;
            int numHiddenB = 5;
            int numOutput = 3;

            DnnStaticTwoHl dnn = new DnnStaticTwoHl(numInput, numHiddenA, numHiddenB, numOutput);

            double[] weights = new double[] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
        0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
        0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
        0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,
        0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59 };

            dnn.SetWeights(weights);

            double[] xValues = new double[] { 1.0, 2.0, 3.0 };

            Console.WriteLine("\nDummy weights and bias values are:");
            ShowVector(weights, 10, 2, true);

            Console.WriteLine("\nDummy inputs are:");
            ShowVector(xValues, 3, 1, true);

            double[] yValues = dnn.ComputeOutputs(xValues);

            Console.WriteLine("\nComputed outputs are:");
            ShowVector(yValues, 3, 4, true);


            Console.WriteLine("\nEnd deep neural network input-output demo\n");
            Console.ReadLine();
        }
        public static void testDNNTraining()
        {
            Console.WriteLine("\nBegin DNN with back-propagation demo");
            DateTime startTime = DateTime.Now;
            int numInput = 4;
            int numHiddenA = 1;//5;
            int numHiddenB = 5;//5;
            int numOutput = 3;
            int numRows = 1000;
            int seed = 1; // gives nice demo

            Console.WriteLine("\nGenerating " + numRows +
              " artificial data items with " + numInput + " features");
            double[][] allData = MakeAllData(numInput, numHiddenA,numHiddenB, numOutput,
              numRows, seed);
            Console.WriteLine("Done");

            //ShowMatrix(allData, allData.Length, 2, true);

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            SplitTrainTest(allData, 0.80, seed, out trainData, out testData);
            Console.WriteLine("Done\n");

            Console.WriteLine("Training data:");
            ShowMatrix(trainData, 4, 2, true);
            Console.WriteLine("Test data:");
            ShowMatrix(testData, 4, 2, true);

            Console.WriteLine("Creating a " + numInput + "-" + numHiddenA + "-" + numHiddenB +
              "-" + numOutput + " neural network");
            DnnStaticTwoHl nn = new DnnStaticTwoHl(numInput, numHiddenA,numHiddenB, numOutput);

            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;
            Console.WriteLine("Setting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F2"));
            Console.WriteLine("Setting momentum  = " + momentum.ToString("F2"));

            Console.WriteLine("\nStarting training");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("Done");
            //Console.WriteLine("\nFinal neural network model weights and biases:\n");
            //ShowVector(weights, 2, 10, true);

            //double[] y = nn.ComputeOutputs(new double[] { 1.0, 2.0, 3.0, 4.0 });
            //ShowVector(y, 3, 3, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nFinal accuracy on training data = " +
              trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("Final accuracy on test data     = " +
              testAcc.ToString("F4"));

            TimeSpan runTime = DateTime.Now - startTime;
            Console.WriteLine("\nTime elapsed: " + Convert.ToString(runTime.TotalMilliseconds / 1000) + " s");
            Console.WriteLine("\nEnd DNN with back-propagation demo\n");
            Console.ReadLine();
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

        static double[][] MakeAllData(int numInput, int numFirstHidden, int numSecondHidden,
          int numOutput, int numRows, int seed)
        {
            Random rnd = new Random(seed);
            int numWeights = (numInput * numFirstHidden) + numFirstHidden + (numFirstHidden * numSecondHidden) +
                numSecondHidden + (numSecondHidden * numOutput) + numOutput;

            double[] weights = new double[numWeights]; // actually weights & biases
            for (int i = 0; i < numWeights; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to 10.0]

            Console.WriteLine("Generating weights and biases:");
            ShowVector(weights, 2, 10, true);

            double[][] result = new double[numRows][]; // allocate return-result
            for (int i = 0; i < numRows; ++i)
                result[i] = new double[numInput + numOutput]; // 1-of-N in last column

            DnnStaticTwoHl gnn =
              new DnnStaticTwoHl(numInput, numFirstHidden,numSecondHidden, numOutput); // generating NN
            gnn.SetWeights(weights);

            for (int r = 0; r < numRows; ++r) // for each row
            {
                // generate random inputs
                double[] inputs = new double[numInput];
                for (int i = 0; i < numInput; ++i)
                    inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to -10.0]

                // compute outputs
                double[] outputs = gnn.ComputeOutputs(inputs);

                // translate outputs to 1-of-N
                double[] oneOfN = new double[numOutput]; // all 0.0

                int maxIndex = 0;
                double maxValue = outputs[0];
                for (int i = 0; i < numOutput; ++i)
                {
                    if (outputs[i] > maxValue)
                    {
                        maxIndex = i;
                        maxValue = outputs[i];
                    }
                }
                oneOfN[maxIndex] = 1.0;

                // place inputs and 1-of-N output values into curr row
                int c = 0; // column into result[][]
                for (int i = 0; i < numInput; ++i) // inputs
                    result[r][c++] = inputs[i];
                for (int i = 0; i < numOutput; ++i) // outputs
                    result[r][c++] = oneOfN[i];
            } // each row
            return result;
        } // MakeAllData

        static void SplitTrainTest(double[][] allData, double trainPct,
          int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct); // usually 0.80
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } // SplitTrainTest

        public static void ShowMatrix(double[][] matrix, int numRows,
         int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices == true)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }
    }
}
