using System;
using System.IO;
using System.Collections.Generic;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Regression;
using SharpLearning.InputOutput.Csv;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.Optimization;
using RDotNet;

namespace machinelearning
{
    public static class Learning
    {   
        public static double FitGBT(double[] pred_Features)
        {
   
            var parser = new CsvParser(() => new StreamReader("dataset.csv"),separator:',');
            var targetName = "Y";

			var observations = parser.EnumerateRows(c => c != targetName)
		    .ToF64Matrix();

			var targets = parser.EnumerateRows(targetName)
				.ToF64Vector();
            
            // read regression targets

            
            var metric = new MeanSquaredErrorRegressionMetric();

            var parameters = new double[][]
            {
                new double[] { 80, 300 }, // iterations (min: 20, max: 100)
                new double[] { 0.02, 0.2 }, // learning rate (min: 0.02, max: 0.2)
                new double[] { 8, 15 }, // maximumTreeDepth (min: 8, max: 15)
                new double[] { 0.5, 0.9 }, // subSampleRatio (min: 0.5, max: 0.9)
                new double[] { 1,  observations.ColumnCount}, // featuresPrSplit (min: 1, max: numberOfFeatures)
            };


            var validationSplit = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
                                                                             .SplitSet(observations, targets);

            Func<double[], OptimizerResult> minimize = p =>
            {
                // create the candidate learner using the current optimization parameters

                var candidateLearner = new RegressionSquareLossGradientBoostLearner(
                                     iterations: (int)p[0],
                                     learningRate: p[1],
                                     maximumTreeDepth: (int)p[2],
                                     subSampleRatio: p[3],
                                     featuresPrSplit: (int)p[4],
                                     runParallel: false);

                var candidateModel = candidateLearner.Learn(validationSplit.TrainingSet.Observations,
                validationSplit.TrainingSet.Targets);

                var validationPredictions = candidateModel.Predict(validationSplit.TestSet.Observations);
                var candidateError = metric.Error(validationSplit.TestSet.Targets, validationPredictions);

                return new OptimizerResult(p, candidateError);
            };

            // Hyper-parameter tuning 
            var optimizer = new RandomSearchOptimizer(parameters, iterations: 30, runParallel: true);

            var result = optimizer.OptimizeBest(minimize);
            var best = result.ParameterSet;

            var learner = new RegressionSquareLossGradientBoostLearner(
                                iterations: (int)best[0],
                                learningRate: best[1],
                                maximumTreeDepth: (int)best[2],
                                subSampleRatio: best[3],
                                featuresPrSplit: (int)best[4],
                                runParallel: false);

            var model = learner.Learn(observations, targets);
            var prediction = model.Predict(pred_Features);

            return prediction;
        }


        public static double FitEN(double[] pred_Features)
        {
            ///////////////////////
            /// 
            /// Using Elastic Net algorithm in R and 
            /// Tuning parameters via MLR library in R 
            /// 
            /// //////////////////

            SaveArrayAsCSV(pred_Features, "pred_Features.csv");

            REngine.SetEnvironmentVariables();
            REngine engin = REngine.GetInstance();
            
            engin.Evaluate(@"

library(mlr)
library(glmnet)

dataset <- read.csv('dataset.csv',header =  TRUE)
pred_features <- read.csv('pred_Features.csv', header =FALSE)
regTask <- makeRegrTask(id= 'Reg.EN', data = dataset, target = 'Y')
ENlearner <- makeLearner('regr.glmnet')

PS <- makeParamSet(
  makeNumericParam('alpha', lower =0, upper =1),
  makeNumericParam('lambda', lower =0, upper =0.25)
)

rdesc = makeResampleDesc('Holdout')
Tuning <- makeTuneControlIrace(maxExperiments = 200L)
res <-tuneParams(ENlearner, task=regTask,par.set=PS,control=Tuning,resampling=rdesc,measures = list(mse),show.info = FALSE)
ENlearner <- setHyperPars(ENlearner,  par.vals = res$x)
model <- train(learner = ENlearner, task = regTask)
colnames(pred_features) <- names(dataset)
prediction <- predict(model, newdata = pred_features)

");
            
            double prediction = engin.Evaluate("prediction$data[2]$response").AsNumeric().ToArray()[0];
            return prediction;
        }


        public static float[][] Trans_x(double[,] X)
        {

			var dim0 = X.GetLength(0);
			var dim1 = X.GetLength(1);


            float[][] X_trans = new float[dim0][];

			for (int i = 0; i < dim0; i++)
			{
                X_trans[i] = new float[dim1];

				for (int j = 0; j < dim1; j++)
				{
                      X_trans[i][j] = (float) X[i, j];
                }
			}

            return X_trans;
		}

        public static float[] Trans_y(double[,] X)
		{

			var dim0 = X.GetLength(0);
			var dim1 = X.GetLength(1);


            float[] X_trans = new float[dim0];

			for (int i = 0; i < dim0; i++)
			{

                if (X[i, 0] > 0 )
                {
                    X_trans[i]=1f;
                }

                else
                {
                    X_trans[i] = 0f;
                }

            }

			return X_trans;
		}


		public static float GetRandomNumber(double minimum, double maximum)
		{
			Random random = new Random();
            var input = random.NextDouble() * (maximum - minimum) + minimum;

			float result = (float)input;
			if (float.IsPositiveInfinity(result))
			{
				result = float.MaxValue;
			}
			else if (float.IsNegativeInfinity(result))
			{
				result = float.MinValue;
			}

            return result ;
		}

        public static void SaveArrayAsCSV<T>(T[] arrayToSave, string fileName)
        {
            using (StreamWriter file = new StreamWriter(fileName))
            {
                foreach (T item in arrayToSave)
                {
                    file.Write(item + ",");
                }
            }
        }

    }
}