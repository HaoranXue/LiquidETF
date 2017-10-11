using System;
using System.IO;
using System.Collections.Generic;
using RDotNet;
using Preprocessing;
using Deedle;


namespace portfolio_optimization

{
    public static class PO
    {
        public static double[] ETF2Allocation(List<string> ETFs,  DataPreProcessing pro)
        {
            // Get the history data 

            List<Series<DateTime, double>> ETF_Hisc = new List<Series<DateTime, double>>() ;

            for (int i = 0; i < ETFs.Count; i++)
            {
                string ETFname = ETFs[i];

                for (int j = 0; j < pro.Trade_ETF.Count; j++)
                {
                    if (ETFname == pro.Trade_ETF[j])
                    {

                        ETF_Hisc.Add(pro.ETF_list[j]);

                    }
                }
            }

            // Transfer list<series> to array 

            double[][] ETF_Hisc_arrary = new double[ETF_Hisc.Count][];

            for (int i = 0; i < ETF_Hisc.Count; i++)
            {   
                
                var len = ETF_Hisc[i].ValueCount;
                ETF_Hisc_arrary[i] = new double[len];

                for (int j = 0; j < len; j++)
                {
                    ETF_Hisc_arrary[i][j] = ETF_Hisc[i].GetAt(j);
                
                }

            }


			Random MC = new Random();

			double[][] ETF_mc_array = new double[5][];

			for (int i = 0; i < 5; i++)
			{
				ETF_mc_array[i] = new double[1000];
			}

			for (int i = 0; i < 1000; i++)
			{
				int index = MC.Next(0, 99);

				for (int j = 0; j < 5; j++)
				{
					ETF_mc_array[j][i] = ETF_Hisc_arrary[j][index];
				}
			}

			SaveArrayAsCSV(ETF_mc_array, "Hisc_array.csv");

            // Optimization 

            REngine.SetEnvironmentVariables(); // <-- May be omitted; the next line would call it.
            REngine engine = REngine.GetInstance();

			// Import library
			engine.Evaluate(@"

library(PortfolioAnalytics)
library(DEoptim)
data <- t(read.csv('Hisc_array.csv',header = FALSE))[-1001,]
rownames(data) <- seq(1,1000,1)
colnames(data) <- c('x1','x2','x3','x4','x5')
portfolio <- portfolio.spec(colnames(data))
portfolio <- add.constraint(portfolio,  type = 'weight_sum',min_sum=0.99, max_sum=1.01)
portfolio <- add.constraint(portfolio, type = 'box', min = 0.1, max = 0.3)
portfolio <- add.objective(portfolio=portfolio, type='risk', name='VaR',arguments = list(p = 0.97, method = 'historical',portfolio_method='component'), enabled = TRUE)
result<- optimize.portfolio(as.ts(data), portfolio = portfolio,traceDE=5,optimize_method='DEoptim',search_size=2000)

");
			// Caculate the results 
			double[] allocations = engine.Evaluate("result$weights").AsNumeric().ToArray();

			return allocations;
          
        }

        public static double[] ETF2AllocationOwnOptim(List<string> ETFs, DataPreProcessing pro)
        {
            List<Series<DateTime, double>> ETF_Hisc = new List<Series<DateTime, double>>();

            for (int i = 0; i < ETFs.Count; i++)
            {
                string ETFname = ETFs[i];

                for (int j = 0; j < pro.Trade_ETF.Count; j++)
                {
                    if (ETFname == pro.Trade_ETF[j])
                    {

                        ETF_Hisc.Add(pro.Optimizing_data[j]);

                    }
                }
            }

            // Transfer list<series> to array 

            double[][] ETF_Hisc_arrary = new double[ETF_Hisc.Count][];

            for (int i = 0; i < ETF_Hisc.Count; i++)
            {

                var len = ETF_Hisc[i].ValueCount;
                ETF_Hisc_arrary[i] = new double[len];

                for (int j = 0; j < len; j++)
                {
                    ETF_Hisc_arrary[i][j] = ETF_Hisc[i].GetAt(j);

                }

            }


            Random MC = new Random();

            double[][] ETF_mc_array = new double[5][];

            for (int i = 0; i < 5; i++)
            {
                ETF_mc_array[i] = new double[1000];
            }

            for (int i = 0; i < 1000; i++)
            {
                int index = MC.Next(0, 99);

                for (int j = 0; j < 5; j++)
                {
                    ETF_mc_array[j][i] = ETF_Hisc_arrary[j][index];
                }
            }

            SaveArrayAsCSV(ETF_mc_array, "Hisc_array.csv");

            // Optimization 

            REngine.SetEnvironmentVariables(); // <-- May be omitted; the next line would call it.
            REngine engine = REngine.GetInstance();

            // Import library
            engine.Evaluate(@"

library(nloptr)

data <- t(read.csv('Hisc_array.csv',header = FALSE))[-1001,]

numberAssets <- length(data[1,])
numberDays <- length(data[,1])

fn <- function(w)
{
portfolio <- rep(0,numberDays)
for(i in 1:numberAssets)
{
portfolio <- portfolio + data[,i]*w[i]
}
target <-  -  mean(portfolio)  / quantile(portfolio,0.025)
return (target)
}
he <- function(w) sum(w) -1   
w <- rep(0.2,numberAssets)
result <- auglag(x0 = w, fn = fn, lower = rep(0.1,numberAssets), upper = rep(0.3,numberAssets), heq = he, localsolver = 'SLSQP')
print(result$par)
");     

            double[] allocations = engine.Evaluate("result$par").AsNumeric().ToArray();

            return allocations;


        }


		public static double[] ETF2AllocationD(List<string> ETFs, DataPreProcessing pro)
		{
			// Get the history data 

			List<Series<DateTime, double>> ETF_Hisc = new List<Series<DateTime, double>>();

			for (int i = 0; i < ETFs.Count; i++)
			{
				string ETFname = ETFs[i];

				for (int j = 0; j < pro.Trade_ETF.Count; j++)
				{
					if (ETFname == pro.Trade_ETF[j])
					{

                        ETF_Hisc.Add(pro.Optimizing_data[j]);

					}
				}
			}

			// Transfer list<series> to array 

			double[][] ETF_Hisc_arrary = new double[ETF_Hisc.Count][];

			for (int i = 0; i < ETF_Hisc.Count; i++)
			{

				var len = ETF_Hisc[i].ValueCount;
				ETF_Hisc_arrary[i] = new double[len];

				for (int j = 0; j < len; j++)
				{
					ETF_Hisc_arrary[i][j] = ETF_Hisc[i].GetAt(j);

				}

			}


			Random MC = new Random();

			double[][] ETF_mc_array = new double[5][];

			for (int i = 0; i < 5; i++)
			{
				ETF_mc_array[i] = new double[1000];
			}

			for (int i = 0; i < 1000; i++)
			{
				int index = MC.Next(0, 99);

				for (int j = 0; j < 5; j++)
				{
					ETF_mc_array[j][i] = ETF_Hisc_arrary[j][index];
				}
			}

			SaveArrayAsCSV(ETF_mc_array, "Hisc_array.csv");

			// Optimization 

			REngine.SetEnvironmentVariables(); // <-- May be omitted; the next line would call it.
			REngine engine = REngine.GetInstance();

			// Import library
			engine.Evaluate(@"

library(PortfolioAnalytics)
library(DEoptim)
data <- t(read.csv('Hisc_array.csv',header = FALSE))[-1001,]
rownames(data) <- seq(1,1000,1)
colnames(data) <- c('x1','x2','x3','x4','x5')
portfolio <- portfolio.spec(colnames(data))
portfolio <- add.constraint(portfolio,  type = 'weight_sum',min_sum=0.99, max_sum=1.01)
portfolio <- add.constraint(portfolio, type = 'box', min = 0.1, max = 0.3)
portfolio <- add.objective(portfolio=portfolio, type='risk', name='VaR',arguments = list(p = 0.97, method = 'historical',portfolio_method='component'), enabled = TRUE)
result<- optimize.portfolio(as.ts(data), portfolio = portfolio,traceDE=5,optimize_method='DEoptim',search_size=2000)

");
			// Caculate the results 
			double[] allocations = engine.Evaluate("result$weights").AsNumeric().ToArray();

			return allocations;

		}



		public static void SaveArrayAsCSV<T>(T[][] jaggedArrayToSave, string fileName)
        {
            using (StreamWriter file = new StreamWriter(fileName))
            {
                foreach (T[] array in jaggedArrayToSave)
                {
                    foreach (T item in array)
                    {
                        file.Write(item + ",");
                    }
                    file.Write(Environment.NewLine);
                }
            }
        }


    }
}
