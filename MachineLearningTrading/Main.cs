using System;
using System.Collections.Generic;
using System.Collections;
using System.IO;
using Deedle;
using MathNet.Numerics.Statistics;
using machinelearning;
using portfolio_optimization;
using Preprocessing;
using BacktestSystem;
using System.Linq;


namespace MLtrading
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Directory.SetCurrentDirectory("C:/Users/Haoran/Documents/Visual Studio 2015/Projects/LiquidETFproject/MachineLearningTrading");

            Console.WriteLine("Enter 0 for backtest the strategy, enter 1 for trading via the strategy");
            string Order = Console.ReadLine();

            if (Order == "0")
            {
                Console.WriteLine("Please enter the start date you want to do the backtest(Format: YYYY-MM-DD)");
                string Date = Console.ReadLine();
                Console.WriteLine("Please enter the weeks you want to do the backtest");
                string Weeks = Console.ReadLine();

                BackTest(Date, Weeks);
            }
            else if (Order == "1")
            {
                Trade();
            }
            else
            {
                Console.WriteLine("Error of setting or backtet. ");
            }

            Console.ReadKey();
        }

        public static void BackTest(string Date, string Weeks)
        {


            ///////////////////////////
            /// 
            /// Setting backtest object
            /// 
            //////////////////////////

            Backtest Mybacktest = new Backtest();
            Backtest Mybacktest_adj = new Backtest();

            // Initial the two backtest 

            Mybacktest.Init();
            Mybacktest_adj.Init();

            ///////////////////////////
            /// 
            /// Setting data container to store the information
            /// outside the for loop 
            /// 
            ///////////////////////////

            // setting netvalue lists for both standard strategy and dynamic strategy

            List<double> Hisc_netValue = new List<double>();
            List<double> Adj_netValue = new List<double>();

            // setting lists to store Fixed Income ETFs and Equity ETFs
            // we are holding during this trading week for the caculation of 
            // Turnover Utility function

            List<string> ETFs_holding_FI = new List<string>();
            List<string> ETFs_holding_Equ = new List<string>();

            // setting matrix to store the trading history during this backtest

            string[][] trading_history_ETF = new string[Convert.ToInt64(Weeks)][];
            double[][] trading_history_allocation = new double[Convert.ToInt64(Weeks)][];
            double[][] ADJtrading_history_allocation = new double[Convert.ToInt64(Weeks)][];

            // setting two variable drawdown and position ratio for the caculation
            // in Dynamic strategy, position ratio means the percentage of Fixed Income 
            // ETFs we are currently holding

            double DrawDown = 0;
            double Position_ratio = 0.2;

            int FI_holding_weeks = 0;
            int Equ_holding_weeks = 0;

            double[] FI_holding_allocation = new double[5];
            double[] Equ_holding_allocation = new double[5];

            //////////////////////////////////
            /// 
            /// For loop backtest 
            /// 
            /////////////////////////////////


            for (int i = 0; i < Convert.ToInt64(Weeks); i++)
            {

                // seting Datapreprocessing class for both Fixed Income ETFs and Equity ETFs

                DataPreProcessing pro_FI = new DataPreProcessing();
                DataPreProcessing pro_Equ = new DataPreProcessing();

                // caculate the date of today start from the 'Date'
                // which is given by backtest function

                var Today = DateTime.Parse(Date).AddDays(i * 7);

                // print the date we trained the model and trade 

                Console.WriteLine("Date: {0}", Today.ToString());

                // cleaning data use data preprocessing class 

                pro_FI.Run(Today.ToString(), 112, "Fixed Income");
                pro_Equ.Run(Today.ToString(), 112, "Equity");

                // Set prediction vector

                double[] predictions_FI = new double[pro_FI.Trade_ETF.Count];
                double[] predictions_Equ = new double[pro_Equ.Trade_ETF.Count];

                // Set blend ETFs list to store the Top 10 etfs which is going to be
                // longed by the algorithm

                List<string> Blend_ETFs = new List<string>();

                /////////////////////////////
                ///                   ///////
                /// FI ETF prediction ///////
                ///                   ///////
                /////////////////////////////

                for (int j = 0; j < pro_FI.Trade_ETF.Count; j++)
                {

                    // Grab the data from the datapreprocessing object class

                    var y = pro_FI.Target_List[j];
                    var fy = new FrameBuilder.Columns<DateTime, string>{
                     { "Y"  , y  }
                    }.Frame;
                    var data = pro_FI.Feature_List[j].Join(fy);
                    var pred_Features = pro_FI.pred_Feature_List[j];

                    data.SaveCsv("dataset.csv");

                    // Training machine learning and predict

                    var prediction = Learning.FitGBT(pred_Features);

                    predictions_FI[j] = prediction;

                }

                // Get the minimum scores of top 5 ETF 

                var hold_FI = PredRanking(predictions_FI, 5);

                // Get the namelist of top 5 ETF

                List<string> ETFs_FI = new List<string>();

                for (int m = 0; m < pro_FI.Trade_ETF.Count; m++)
                {
                    if (predictions_FI[m] >= hold_FI)
                    {
                        ETFs_FI.Add(pro_FI.Trade_ETF[m]);
                    }
                }


                // Caculate the bid-ask spread cost if trade all 5 ETFs given by algorithm

                double[] FixedIncomeSpread = new double[5];
                FixedIncomeSpread = GetBidAskSpread(ETFs_FI.ToArray());

                // Cacualte the Unitility and decide if we should trade this week

                if (i == 0)
                {
                    ETFs_holding_FI = ETFs_FI;
                }
                else
                {
                    // get all prediction results of both holding etf and etfs which we may be going to trade

                    double[] holding_pred = ETFname2Prediction(ETFs_holding_FI, predictions_FI, pro_FI);
                    double[] long_pred = ETFname2Prediction(ETFs_FI, predictions_FI, pro_FI);

                    // caculate the trade diff which is the utility of trading this week

                    double trade_diff = long_pred.Sum() -
                                                 holding_pred.Sum() -
                                                 CaculateRebalanceCost(ETFs_FI,
                                                                       ETFs_holding_FI,
                                                                       FI_holding_allocation,
                                                                       pro_FI);

                    // check if it is worth of trading

                    if (trade_diff < 0)
                    {
                        // It is not worth of trading and ETFs portfolio will be same

                        FI_holding_weeks += 1;
                        ETFs_FI = ETFs_holding_FI;

                        // setting spread equals to 0 because we are not going to trade this week

                        FixedIncomeSpread = new double[] { 0, 0, 0, 0, 0 };
                    }
                    else
                    {
                        FI_holding_weeks = 0;

                        // It is worth of changing positions and trading this week ! 

                        // recaculate the spread costs because we may not going to trade all 
                        // ETFs which we are holding right now. 

                        for (int m = 0; m < 5; m++)
                        {
                            for (int n = 0; n < 5; n++)
                            {
                                if (ETFs_FI[m] == ETFs_holding_FI[n])
                                {
                                    FixedIncomeSpread[m] = 0;
                                }
                                else
                                {
                                    continue;
                                }
                            }
                        }

                        // resetting the fixed income ETFs  we are holding 

                        ETFs_holding_FI = ETFs_FI;
                    }

                }

                Console.WriteLine("Long the following ETFs: ");

                // Store the Fixed Income ETFs namelist to blend ETFs list

                for (int n = 0; n < ETFs_FI.Count; n++)
                {
                    Console.WriteLine(ETFs_FI[n]);
                    Blend_ETFs.Add(ETFs_FI[n]);
                }

                ///////////////////////////////////
                ///                         ///////
                /// Equity ETF prediction   ///////
                ///                         ///////
                ///////////////////////////////////


                for (int j = 0; j < pro_Equ.Trade_ETF.Count; j++)
                {
                    // Run machine learning and predict next week for all ETFss

                    var y = pro_Equ.Target_List[j];

                    var fy = new FrameBuilder.Columns<DateTime, string>{
                     { "Y"  , y  }
                    }.Frame;
                    var data = pro_Equ.Feature_List[j].Join(fy);
                    var pred_Features = pro_Equ.pred_Feature_List[j];

                    data.SaveCsv("dataset.csv");

                    var prediction = Learning.FitGBT(pred_Features);

                    predictions_Equ[j] = prediction;
                }


                List<string> ETFs_Equ = new List<string>();

                // Find the min score of top 5 best ETFs

                var hold_Equ = PredRanking(predictions_Equ, 5);

                for (int m = 0; m < pro_Equ.Trade_ETF.Count; m++)
                {
                    if (predictions_Equ[m] >= hold_Equ)
                    {
                        ETFs_Equ.Add(pro_Equ.Trade_ETF[m]);
                    }
                }

                // caculate the bidAsk Spread

                double[] EquityBASpread = new double[5];
                EquityBASpread = GetBidAskSpread(ETFs_Equ.ToArray());

                if (i == 0)
                {
                    ETFs_holding_Equ = ETFs_Equ;
                }
                else
                {

                    double[] holding_pred = ETFname2Prediction(ETFs_holding_Equ, predictions_Equ, pro_Equ);
                    double[] long_pred = ETFname2Prediction(ETFs_Equ, predictions_Equ, pro_Equ);

                    // Caculate the Utility

                    double trade_diff = long_pred.Sum() -
                                                 holding_pred.Sum() -
                                                    CaculateRebalanceCost(ETFs_Equ,
                                                                             ETFs_holding_Equ,
                                                                             Equ_holding_allocation,
                                                                             pro_Equ);

                    // check if it is worth of trading this week

                    if (trade_diff < 0)
                    {
                        Equ_holding_weeks += 1;
                        ETFs_Equ = ETFs_holding_Equ;
                        EquityBASpread = new double[] { 0, 0, 0, 0, 0 };
                    }
                    else
                    {
                        // Recacluate the spread costs

                        Equ_holding_weeks = 0;

                        for (int m = 0; m < 5; m++)
                        {
                            for (int n = 0; n < 5; n++)
                            {
                                if (ETFs_Equ[m] == ETFs_holding_Equ[n])
                                {
                                    EquityBASpread[m] = 0;
                                }
                                else
                                {
                                    continue;
                                }
                            }
                        }

                        ETFs_holding_Equ = ETFs_Equ;
                    }

                }

                // Store the Equity ETFs we are going to long in Blend ETFs list

                for (int n = 0; n < ETFs_Equ.Count; n++)
                {
                    Console.WriteLine(ETFs_Equ[n]);
                    Blend_ETFs.Add(ETFs_Equ[n]);
                }

                //  Caculate optimized allocations for both Fixed income and Equity ETFs

                //////////////////////////////

                Console.WriteLine("Holding weeks for current Fixed Income ETFs is {0}", FI_holding_weeks);
                Console.WriteLine("Holding weeks for current Equity ETFs is {0}", Equ_holding_weeks);

                double[] AllocationFI = new double[5];

                if (FI_holding_weeks == 0)
                {
                    AllocationFI = PO.ETF2AllocationOwnOptim(ETFs_FI, pro_FI);
                    FI_holding_allocation = AllocationFI;
                }
                else if (FI_holding_weeks < 15)
                {
                    AllocationFI = FI_holding_allocation;
                }
                else
                {
                    FI_holding_weeks = 0;
                    AllocationFI = PO.ETF2AllocationOwnOptim(ETFs_FI, pro_FI);
                    FI_holding_allocation = AllocationFI;
                }

                //////////////////////////////

                double[] AllocationEqu = new double[5];

                if (Equ_holding_weeks == 0)
                {
                    AllocationEqu = PO.ETF2AllocationOwnOptim(ETFs_Equ, pro_Equ);
                    Equ_holding_allocation = AllocationEqu;
                }
                else if (Equ_holding_weeks < 15)
                {
                    AllocationEqu = Equ_holding_allocation;
                }
                else
                {
                    Equ_holding_weeks = 0;
                    AllocationEqu = PO.ETF2AllocationOwnOptim(ETFs_Equ, pro_Equ);
                    Equ_holding_allocation = AllocationEqu;
                }

                //////////////////////////////

                // Setting allocations which is an array to store allocations for strandard strategy

                double[] allocations = new double[10];

                for (int fi = 0; fi < 5; fi++)
                {
                    allocations[fi] = AllocationFI[fi] * 0.2;
                }

                for (int equ = 0; equ < 5; equ++)
                {
                    allocations[equ + 5] = AllocationEqu[equ] * 0.8;
                }


                // Setting ALLOCATION which is an array to store allocations for strandard strategy

                double[] ALLOCATION = new double[10];

                for (int fi = 0; fi < 5; fi++)
                {
                    ALLOCATION[fi] = AllocationFI[fi] * Position_ratio;
                }

                for (int equ = 0; equ < 5; equ++)
                {
                    ALLOCATION[equ + 5] = AllocationEqu[equ] * (1 - Position_ratio);
                }

                //  Transform ETFs list to an array 

                string[] ETFs = new string[10];

                for (int etf = 0; etf < 10; etf++)
                {
                    ETFs[etf] = Blend_ETFs[etf];
                }

                // Storing the ETFs trading history and allocations

                trading_history_ETF[i] = new string[10];
                trading_history_ETF[i] = ETFs;

                trading_history_allocation[i] = new double[10];
                trading_history_allocation[i] = allocations;

                ADJtrading_history_allocation[i] = new double[10];
                ADJtrading_history_allocation[i] = ALLOCATION;

                // Get the spread array for all ETFs

                double[] spread = new double[10];

                Array.Copy(FixedIncomeSpread, spread, FixedIncomeSpread.Length);
                Array.Copy(EquityBASpread, 0, spread, FixedIncomeSpread.Length, EquityBASpread.Length);

                // Caculate the weighted spread for the adjustment in netvalue

                double weighted_spread = 0;

                for (int spreadItem = 0; spreadItem < 10; spreadItem++)
                {
                    weighted_spread += allocations[spreadItem] * spread[spreadItem];
                }

                // clearing the NetValue

                Hisc_netValue.Add(Mybacktest.Rebalance(Today, ETFs, allocations) * (1 - weighted_spread));
                Adj_netValue.Add(Mybacktest_adj.Rebalance(Today, ETFs, ALLOCATION) * (1 - weighted_spread));

                // Caculate the current drawdown and adjust the position ratio which is 
                //  the percentage for the fixed income ETFs

                if (i == 0)
                {
                    DrawDown = 0;
                }
                else
                {
                    DrawDown = 1 - Hisc_netValue.Last() / Hisc_netValue.Max();
                }

                // Adjust the position ratio

                if (DrawDown > 0.1)
                {
                    Position_ratio = 0.8;
                }
                else if (DrawDown > 0.08)
                {
                    Position_ratio = 0.6;
                }
                else if (DrawDown > 0.05)
                {
                    Position_ratio = 0.4;
                }
                else
                {
                    Position_ratio = 0.2;
                }

                // Print out the current drawdown and position ratio.

                Console.WriteLine("Current drawdown is: {0}", DrawDown);
                Console.WriteLine("Fixed Income has been adjusted to: {0} %", Position_ratio * 100);

            }


            // Result analysis for NetValue

            /////////////////////
            /// 
            /// Backtest Metrics of Strandard Strategy
            /// 
            ////////////////////


            var StrategyNetValue = Hisc_netValue.ToArray();

            double MaxDD = 0;

            for (int i = 1; i < StrategyNetValue.Length; i++)
            {
                var MaxNetValue = StrategyNetValue.Take(i).Max();
                double drawdown = 1 - StrategyNetValue[i] / MaxNetValue;

                if (drawdown > MaxDD)
                {
                    MaxDD = drawdown;
                }
            }

            Console.WriteLine("Maximum drawdown of This Strategy is: {0}", MaxDD);

            var AnnualReturn = Math.Log(StrategyNetValue.Last()) /
                                                        (Convert.ToDouble(Weeks) / 50);

            Console.WriteLine("Annual Return of This Strategy is: {0}", AnnualReturn);

            var StrategyReturn = NetValue2Return(StrategyNetValue);
            Console.WriteLine("Standard Deviation of This Strategy is: {0}",
                                      Statistics.StandardDeviation(StrategyReturn));

            double[] BTmetrics = new double[3];
            BTmetrics[0] = AnnualReturn;
            BTmetrics[1] = MaxDD;
            BTmetrics[2] = Statistics.StandardDeviation(StrategyReturn) * Math.Sqrt(50);

            // Result analysis for AdjNetValue

            /////////////////////
            /// 
            /// Backtest Metrics of Dynamic Strategy
            /// 
            /////////////////////

            var ADJStrategyNetValue = Adj_netValue.ToArray();

            double ADJMaxDD = 0;

            for (int i = 1; i < ADJStrategyNetValue.Length; i++)
            {
                var MaxNetValue = ADJStrategyNetValue.Take(i).Max();
                double drawdown = 1 - ADJStrategyNetValue[i] / MaxNetValue;

                if (drawdown > ADJMaxDD)
                {
                    ADJMaxDD = drawdown;
                }
            }


            Console.WriteLine("Maximum drawdown of ADJ Strategy is: {0}", ADJMaxDD);

            var ADJAnnualReturn = Math.Log(ADJStrategyNetValue.Last()) /
                                                        (Convert.ToDouble(Weeks) / 50);
            Console.WriteLine("Annual Return of ADJ Strategy is: {0}", ADJAnnualReturn);

            var ADJStrategyReturn = NetValue2Return(ADJStrategyNetValue);
            Console.WriteLine("Standard Deviation of This Strategy is: {0}",
                                      Statistics.StandardDeviation(ADJStrategyReturn));

            double[] ADJBTmetrics = new double[3];
            ADJBTmetrics[0] = ADJAnnualReturn;
            ADJBTmetrics[1] = ADJMaxDD;
            ADJBTmetrics[2] = Statistics.StandardDeviation(ADJStrategyReturn) * Math.Sqrt(50);

            // Output all results to CSV
            // Without position adjustment
            SaveArrayAsCSV_(trading_history_allocation, "StrandardTradingHistoryAllocation.csv");
            SaveArrayAsCSV(BTmetrics, "StandardBacktestMetrics.csv");
            SaveArrayAsCSV(StrategyNetValue, "StandardNet_value.csv");
            // With position adjustmnet
            SaveArrayAsCSV_(ADJtrading_history_allocation, "DynamicTradingHistoryAllocation.csv");
            SaveArrayAsCSV(ADJBTmetrics, "DynamicBacktestMetrics.csv");
            SaveArrayAsCSV(ADJStrategyNetValue, "DynamicNetValue.csv");

            SaveArrayAsCSV_<string>(trading_history_ETF, "ETFTradingHistoryforALL.csv");

            Console.ReadKey();

        }



        public static void Trade()
        {
            //////////////////////////
            /// 
            /// Under Construction
            /// 
            /// /////////////////////

        }

        public static double[] NetValue2Return(double[] netvalue)
        {

            double[] NVskip1 = netvalue.Skip(1).ToArray();
            double[] NVTake1 = netvalue.Take(netvalue.GetLength(0) - 1).ToArray();


            double[] Return = new double[NVskip1.Count()];

            for (int i = 0; i < NVskip1.Count(); i++)
            {

                var This_Return = NVskip1[i] / NVTake1[i];
                Return[i] = This_Return;

            }

            return Return;
        }


        public static double[] ETFname2Prediction(List<string> ETF,
                                                    double[] prediction,
                                                    DataPreProcessing Pro)
        {
            double[] PRE = new double[ETF.Count];

            for (int j = 0; j < ETF.Count; j++)
            {
                for (int i = 0; i < Pro.Trade_ETF.Count; i++)
                {
                    if (Pro.Trade_ETF[i] == ETF[j])
                    {
                        PRE[j] = prediction[i];
                    }
                    else
                    {
                        continue;
                    }
                }
            }

            return PRE;
        }

        public static double CaculateTurnOver(List<string> holding_ETFs, List<string> new_ETFs, double Pcent)
        {
            double turnover = 0;
            var Intersection = new_ETFs.Intersect(holding_ETFs);
            turnover += (5 - Intersection.Count()) * Pcent;
            return turnover;
        }


        public static double PredRanking(double[] predictions, int place)
        {
            ArrayList prediction_sort = new ArrayList(predictions);
            prediction_sort.Sort();
            double hold = Convert.ToDouble(prediction_sort[prediction_sort.Count - place]);

            return hold;
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


        public static void SaveArrayAsCSV_<T>(T[][] jaggedArrayToSave, string fileName)
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


        public static double[] GetBidAskSpread(string[] ETFs)
        {
            double[] spread = new double[ETFs.Length];
            Frame<string, string> BidAskSpread = Frame.ReadCsv("data/SpreadDistribution.csv").IndexRows<string>("ETFs");

            for (int etf = 0; etf < ETFs.Length; etf++)
            {
                spread[etf] = BidAskSpread.GetRow<double>(ETFs[etf]).Get("Mean");

            }

            return spread;

        }

        public static double CaculateRebalanceCost(List<string> ETFs,
                                                   List<string> ETFs_holding,
                                                   double[] ETFs_holding_allocation,
                                                   DataPreProcessing preprocessing)
        {
            var Expected_Allocation = PO.ETF2AllocationOwnOptim(ETFs, preprocessing);
            double rebalance = 0;

            var Union_ETFs = ETFs.Union(ETFs_holding).ToArray();

            double[] Lastweek = new double[Union_ETFs.Count()];
            double[] Thisweek = new double[Union_ETFs.Count()];



            for (int j = 0; j < Union_ETFs.Count(); j++)
            {

                Lastweek[j] = 0;

                for (int i = 0; i < ETFs_holding.Count(); i++)
                {
                    if (ETFs_holding[i] == Union_ETFs[j])
                    {
                        Lastweek[j] = ETFs_holding_allocation[i];
                    }
                    else
                    {
                        continue;
                    }
                }

            }


            for (int j = 0; j < Union_ETFs.Count(); j++)
            {

                Thisweek[j] = 0;

                for (int i = 0; i < ETFs.Count(); i++)
                {
                    if (ETFs[i] == Union_ETFs[j])
                    {
                        Thisweek[j] = Expected_Allocation[i];
                    }
                    else
                    {
                        continue;
                    }
                }

            }

            for (int i = 0; i < Union_ETFs.Count(); i++)
            {
                rebalance += Math.Abs(Thisweek[i] - Lastweek[i]);
            }

            double rebalance_cost = rebalance * 0.05;

            Console.WriteLine("Rebalance Cost is {0}", rebalance_cost);

            return rebalance_cost;

        }


    }
}
