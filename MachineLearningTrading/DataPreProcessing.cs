﻿using System;
using System.Linq;
using System.Collections.Generic;
using Deedle;

namespace Preprocessing
{  

    public class DataPreProcessing
    {
        // Lists to store the output data 
		public List<Frame<DateTime, string>> Feature_List = new List<Frame<DateTime, string>>();
		public List<Series<DateTime, double>> Target_List = new List<Series<DateTime, double>>();
		public List<Series<DateTime, double>> ETF_list = new List<Series<DateTime, double>>();
        public List<Series<DateTime, double>> Optimizing_data = new List<Series<DateTime, double>>(); // Data for portfolio optimization
		public List<string> Trade_Index = new List<string>();
		public List<string> Trade_ETF = new List<string>();
        public List<double[]> pred_Feature_List = new List<double[]>();


		public IEnumerable<DateTime> GetDaysBetween(DateTime start, DateTime end)
		{
			for (DateTime i = start; i < end; i = i.AddDays(1))
			{
				yield return i;
			}
		}

        // Main function for DataPreProcessing Class to run pre processing
        public void Run(string date, int weeks,string catagory)
        {

            // Get the namelist of index in the defined catagory
            string[] Index_namelist = Trading_category(catagory);
            // Get the namelist of mapping ETF 
            string[] Mapping_ETF_namelist = Get_mapping_ETF(Index_namelist);
            // Get the historical data of Index 
            var Index_data = Get_DataFromList(Index_namelist, "Index");
            // Get the historical data of ETF
            var ETF_data = Get_DataFromList(Mapping_ETF_namelist, "ETF");
            // Adjust the historical data of index by its based Currency
            var Adj_Index_data = Adjust_index_fx(Index_data, Index_namelist);
            // Adjust the historical data of ETF by its based Currency
            var Adj_ETF_data = Adjust_etf_fx(ETF_data, Mapping_ETF_namelist);
            // Date of running the algorithm 
            var endDate = DateTime.Parse(date);
            // First day of traning set 
            var startDate = endDate.AddDays(-weeks * 7);


            for (int i = 0; i < Index_namelist.Length; i++)
            {

                // Check if the Index exist and if the ETF tradable during this period

                double init_value_Index = Index_data[i].Get(startDate);
                double init_value_ETF = ETF_data[i].Get(startDate);
                // If the net value of ETF and Index equals to 1, it means 
                // ETF or Index didn't exist at the start date. 
                if (init_value_ETF.Equals(1)  || init_value_Index.Equals(1))
                {
                    continue;
                }
                else
                {   // Get the historical time_series of ith Index
                    var raw_y = Adj_Index_data[i].Between(startDate, endDate);
                    // Generate trainning features for Index time-series data and drops sparserows
                    var return_features = MultiLagFeaturesEng(Price2Return(raw_y)
                                                        .Chunk(7)
                                                        .Select(x => x.Value.Sum())).DropSparseRows();
                    
                    var price_features = PriceDataFE(raw_y
                                                     .Chunk(7)
                                                     .Select(x => x.Value.LastValue())).DropSparseRows();

                    var features = return_features.Join(price_features, JoinKind.Inner);

                    // Generate the features which are going to be used for prediction 
                    var return_pred_features = Features_engineering_pred(Price2Return(raw_y)
                                                        .Chunk(7)
                                                        .Select(x => x.Value.Sum()));

                    var price_pred_features = PriceDataFE_pred(raw_y.Chunk(7).Select(x => x.Value.LastValue()));

                    double[] pred_features = new double[return_pred_features.Length+price_pred_features.Length];
                    Array.Copy(return_pred_features, pred_features, return_pred_features.Length);
                    Array.Copy(price_pred_features, 0 ,pred_features, return_pred_features.Length, price_pred_features.Length);

                    // Get the etf Return 
                    var raw_etf = Price2Return(Adj_ETF_data[i].Between(startDate,endDate))
                                                          .Chunk(7)
                                                          .Select(x => x.Value.Sum());
                    // Get the starting date of features matrix because we dropped sparserows before
                    var First_key = features.GetRowKeyAt(0);
                    // Get the target y in this time periods 
                    var y = Price2Return(raw_y).Chunk(7).Select(x => x.Value.Sum()).Between(First_key,endDate);
                    // Get the ETF return in this time periods
                    var etf = raw_etf.Between(First_key, endDate);

                    // Store the processed data in the list 
                    Target_List.Add(y);
                    Feature_List.Add(features);
                    pred_Feature_List.Add(pred_features);
                    ETF_list.Add(etf);
                    Optimizing_data.Add(FilterWeekend(Price2Return(raw_y).Between(First_key, endDate)));
                    Trade_Index.Add(Index_namelist[i]);
                    Trade_ETF.Add(Mapping_ETF_namelist[i]);

                }

            }

        }


        public static String[] Trading_category(String category)
        {   
            // Get the List of Fixed Income

            if (category == "Fixed Income")
            {   
                String[] fi_namelist = { "LECPTREU Index", "BCEX2T Index", "LGCPTRUU Index", "IYJD Index",
                                      "IBOXIG Index","LTITTREU Index","LETSTREU Index","FTFIBGT Index",
                                      "IDCOT1TR Index","BEIG1T Index"};

                return fi_namelist;

            }
            // Get the List of Equity

            else if (category == "Equity")
            {
                var Index_meta = Frame.ReadCsv("data/Index Metadata.csv");
                var TYP = Index_meta.GetColumn<String>("SECURITY_TYP");
                var Ticker = Index_meta.GetColumn<String>("Ticker");
               
                List<String> equ_namelist = new List<string>();

                for (int i = 0; i < 135; i++)
                {
                    if (TYP[i] == "Equity Index")
                    {
                        equ_namelist.Add(Ticker[i]);
                    }
                }

                return equ_namelist.ToArray();
            }

            else
            {   
                Console.WriteLine("Error: Input should be Fixed Income or Equity, Using default fixed income as category.");

                String[] fi_namelist = { "LECPTREU Index", "BCEX2T Index", "LGCPTRUU Index", "IYJD Index",
                                      "IBOXIG Index","LTITTREU Index","LETSTREU Index","FTFIBGT Index",
                                      "IDCOT1TR Index","BEIG1T Index"};

                return fi_namelist;
            }
        }


        public static List<Series<DateTime, double>> Get_DataFromList(String[] namelist, string tag)

        {   // Fuction can read the data from database based on the given name list

            if (tag == "Index")
            {   
                var Data = Frame.ReadCsv("data/Index Returns.csv").IndexRows<DateTime>("Date");
                List<Series<DateTime, double>> List_data = new List<Series<DateTime, double>>();

                for (int i = 0; i < namelist.Length; i++)
                {
                    var data = Data.Rows.Select(x => x.Value.GetAs<double>(namelist[i]));
                    List_data.Add(data);
                }

                return List_data;
            }
            else
            {   
                var Data = Frame.ReadCsv("data/ETF Returns.csv").IndexRows<DateTime>("Date");
                List<Series<DateTime, double>> List_data = new List<Series<DateTime, double>>();

                for (int i = 0; i < namelist.Length; i++)
                {
                    var data = Data.Rows.Select(x => x.Value.GetAs<double>(namelist[i]));
                    List_data.Add(data);
                }

                return List_data;
            }
        }


        public static string[] Get_mapping_ETF(string[] namelist)
        {   
            // Function can get Mapping ETF by given a name list of Index
            
            var Mapping_table = Frame.ReadCsv("data/Mapping_Table.csv");
            var Tracking_ETF = Mapping_table.GetColumn<String>("Best Tracking ETF");
            var Index = Mapping_table.GetColumn<String>("Index");

            string[] ETF_list = new String[namelist.Length];

            for (int i = 0; i < 133; i++)
            {
                var index = Index[i];
                var etf = Tracking_ETF[i];

                for (int j = 0; j < namelist.Length; j++)
                {
                    string Full_name = index + " Index";
                    if (Full_name == namelist[j])
                    {
                        ETF_list[j] = etf;
                    }
                }
            }

            return ETF_list;
        }


        public static List<Series<DateTime, double>> Adjust_index_fx(List<Series<DateTime, double>> data, string[] namelist)

        {   
            //Adjust Index return to GBP based
            
            var FX = Frame.ReadCsv("data/FX Returns.csv").IndexRows<DateTime>("Date").FillMissing(Direction.Forward);
           
            var GBPUSD = FX.Rows.Select(x => x
                                         .Value.GetAs<double>("GBPUSD Curncy"));
            
            var GBPEUR = FX.Rows.Select(x =>
                                        x.Value.GetAs<double>("GBPUSD Curncy") /
                                        x.Value.GetAs<double>("EURUSD Curncy"));
          
            var Index_meta = Frame.ReadCsv("data/Index Metadata.csv");

			List<Series<DateTime, double>> List_data = new List<Series<DateTime, double>>();

			for (int i = 0; i < namelist.Length; i++)
            {
                var meta = Index_meta.Where(index => index.Value.GetAs<string>("Ticker") == namelist[i]);
                var CRNCY = meta.Rows.Select(x => x.Value.GetAs<string>("CRNCY")).GetAt(0);

                if(CRNCY == "USD")
                {
                    var FirstKey = data[i].FirstKey();
                    var LastKey = data[i].LastKey();
                    List_data.Add(data[i] / GBPUSD.Between(FirstKey, LastKey)); 
                }
                else if(CRNCY == "EUR")
                {
                    var FirstKey = data[i].FirstKey();
                    var LastKey = data[i].LastKey();
                    List_data.Add(data[i] / GBPEUR.Between(FirstKey,LastKey));
                }
                else
                {
					List_data.Add(data[i]);
                }

            }

            return List_data;
        }


        public static List<Series<DateTime, double>> Adjust_etf_fx(List<Series<DateTime, double>> data, string[] namelist)
		{   

            //Adjust ETF return to GBP based

			var FX = Frame.ReadCsv("data/FX Returns.csv").IndexRows<DateTime>("Date").FillMissing(Direction.Forward); ;
 
			var GBPUSD = FX.Rows.Select(x =>
										x.Value.GetAs<double>("GBPUSD Curncy"));

			var GBPEUR = FX.Rows.Select(x =>
										x.Value.GetAs<double>("GBPUSD Curncy") /
										x.Value.GetAs<double>("EURUSD Curncy"));

			var ETF_meta = Frame.ReadCsv("data/ETF Metadata.csv");

			List<Series<DateTime, double>> List_data = new List<Series<DateTime, double>>();

			for (int i = 0; i < namelist.Length; i++)
			{
                var meta = ETF_meta.Where(index => index.Value.GetAs<string>("Ticker") == namelist[i]);
				var CRNCY = meta.Rows.Select(x => x.Value.GetAs<string>("CRNCY")).GetAt(0);

				if (CRNCY == "USD")
                {
                    var FirstKey = data[i].FirstKey();
                    var LastKey = data[i].LastKey();
                    List_data.Add(data[i] / GBPUSD.Between(FirstKey,LastKey));
				}
				else if (CRNCY == "EUR")
				{
                    var FirstKey = data[i].FirstKey();
                    var LastKey = data[i].LastKey();
                    List_data.Add(data[i] / GBPEUR.Between(FirstKey,LastKey));
				}
				else
				{
					List_data.Add(data[i]);
				}

			}

			return List_data;
		}


        public static Series<DateTime,double> Price2Return(Series<DateTime, double> data, string Return_type="log")
        {   

            //Transfer the price time-seires to return Time series

            if (Return_type =="log")
            {   
                
				return (data / data.Shift(1)).Log();
            }

                var relative_return = (data / data.Shift(1)).Diff(1);
                return relative_return;

        }


        public static Frame<DateTime,string> MultiLagFeaturesEng(Series<DateTime,double> data)
        {   
            //Featreus Enginerring for multiple different lags
           
            return Features_engineering(data,1)
                .Join(Features_engineering(data,2), JoinKind.Inner)
                .Join(Features_engineering(data,3),JoinKind.Inner);

        }

        public static List<DateTime> Resample_key(string starttime, int weeks)
        {   

            //Generate a list of date by given start time and weeks

            List<DateTime> keys =new List<DateTime>();

            var startDate = DateTime.Parse(starttime);
            keys.Add(startDate);

            for (int i = 1; i < weeks+1; i++)
            {
                var date = startDate.AddDays(7*i);      
                keys.Add(date);
            }

            return keys;
        }


        public Series<DateTime,double> FilterWeekend(Series<DateTime, double> data)
        {
            var FirstKey= data.FirstKey();
            var LastKey = data.LastKey();
            var BD = GetDaysBetween(FirstKey, LastKey).Where(d => d.DayOfWeek != DayOfWeek.Saturday && d.DayOfWeek != DayOfWeek.Sunday);
            var FilteredData = data.GetItems(BD);

            return FilteredData;
        }


        public static Frame<DateTime,string> PriceDataFE(Series<DateTime, double> input)
        {
            int shiftn = 1;    
            
			var data = input.Shift(shiftn);

			var MA3 = data.Window(3).Select(x => x.Value.Mean());
			var MA5 = data.Window(5).Select(x => x.Value.Mean());
			var MA10 = data.Window(10).Select(x => x.Value.Mean());

            var dataMA3 = (data / MA3).Select(x => Math.Log(x.Value));
            var dataMA5 = (data / MA5).Select(x => Math.Log(x.Value));
            var dataMA10 = (data / MA10).Select(x => Math.Log(x.Value));

			var MA3MA5 = (MA3 / MA5).Select(x => Math.Log(x.Value));
			var MA3MA10 = (MA3 / MA10).Select(x => Math.Log(x.Value));
			var MA5MA10 = (MA5 / MA10).Select(x => Math.Log(x.Value));

			var MMP3 = data.Window(3).Select(x => Math.Log( x.Value.Max() / x.Value.Min()));
            var MMP5 = data.Window(5).Select(x => Math.Log(x.Value.Max() / x.Value.Min()));
            var MMP10 = data.Window(10).Select(x => Math.Log(x.Value.Max() / x.Value.Min()));

            string dataMA3column = "PriceMA3" +Convert.ToString(shiftn);
            string dataMA5column = "PriceMA5" + Convert.ToString(shiftn);
            string dataMA10column = "PriceMA10" + Convert.ToString(shiftn);

            string MA3MA5column = "MA3MA5PRICE" + Convert.ToString(shiftn);
            string MA3MA10column = "MA3MA10PRICE" + Convert.ToString(shiftn);
            string MA5MA10column = "MA5MA10PRICE" + Convert.ToString(shiftn);

            string MMP3column = "MMP3" + Convert.ToString(shiftn);
            string MMP5column = "MMP5" + Convert.ToString(shiftn);
            string MMP10column = "MMP10" + Convert.ToString(shiftn);

			var Features = new FrameBuilder.Columns<DateTime, string>{
                {dataMA3column, dataMA3},
                {dataMA5column, dataMA5},
                {dataMA10column, dataMA10},
                {MA3MA5column,MA3MA5},
                {MA3MA10column,MA3MA10},
                {MA5MA10column,MA5MA10},
                {MMP3column, MMP3},
                {MMP5column, MMP5},
                {MMP10column, MMP10}
			}.Frame;

			return Features;

		}

        public static double[] PriceDataFE_pred(Series<DateTime, double> input)
		{
			int shiftn = 0;

			var data = input.Shift(shiftn);

			var MA3 = data.Window(3).Select(x => x.Value.Mean());
			var MA5 = data.Window(5).Select(x => x.Value.Mean());
			var MA10 = data.Window(10).Select(x => x.Value.Mean());

			var dataMA3 = (data / MA3).Select(x => Math.Log(x.Value));
			var dataMA5 = (data / MA5).Select(x => Math.Log(x.Value));
			var dataMA10 = (data / MA10).Select(x => Math.Log(x.Value));

			var MA3MA5 = (MA3 / MA5).Select(x => Math.Log(x.Value));
			var MA3MA10 = (MA3 / MA10).Select(x => Math.Log(x.Value));
			var MA5MA10 = (MA5 / MA10).Select(x => Math.Log(x.Value));

			var MMP3 = data.Window(3).Select(x => Math.Log(x.Value.Max() / x.Value.Min()));
			var MMP5 = data.Window(5).Select(x => Math.Log(x.Value.Max() / x.Value.Min()));
			var MMP10 = data.Window(10).Select(x => Math.Log(x.Value.Max() / x.Value.Min()));

			string dataMA3column = "PriceMA3" + Convert.ToString(shiftn);
			string dataMA5column = "PriceMA5" + Convert.ToString(shiftn);
			string dataMA10column = "PriceMA10" + Convert.ToString(shiftn);

			string MA3MA5column = "MA3MA5PRICE" + Convert.ToString(shiftn);
			string MA3MA10column = "MA3MA10PRICE" + Convert.ToString(shiftn);
			string MA5MA10column = "MA5MA10PRICE" + Convert.ToString(shiftn);

			string MMP3column = "MMP3" + Convert.ToString(shiftn);
			string MMP5column = "MMP5" + Convert.ToString(shiftn);
			string MMP10column = "MMP10" + Convert.ToString(shiftn);

			var Features = new FrameBuilder.Columns<DateTime, string>{
				{dataMA3column, dataMA3},
				{dataMA5column, dataMA5},
				{dataMA10column, dataMA10},
				{MA3MA5column,MA3MA5},
				{MA3MA10column,MA3MA10},
				{MA5MA10column,MA5MA10},
				{MMP3column, MMP3},
				{MMP5column, MMP5},
				{MMP10column, MMP10}
			}.Frame;

			var row_length = Features.RowCount;
			var col_length = Features.ColumnCount;

			double[] delay_1 = Features.GetRowAt<double>(row_length - 1).Values.ToArray<double>();

			double[] features_pred = new double[col_length];

			for (int i = 0; i < col_length; i++)
			{
				features_pred[i] = delay_1[i];
			}

			return features_pred;
		  
		}


        public static Frame<DateTime, string> Features_engineering(Series<DateTime, double> input, int shiftn)
        {   

            //Features Engineering function for training machine learning

            var data = input.Shift(shiftn);

            var data2 = data.Select(x => Math.Pow(x.Value, 2));

            var MA3 = data.Window(3).Select(x => x.Value.Mean());
            var MA5 = data.Window(5).Select(x => x.Value.Mean());
            var MA10 = data.Window(10).Select(x => x.Value.Mean());

            var diffdataMA3 = (data / MA3).Select(x => x.Value - 1);
            var diffdataMA5 = (data / MA5).Select(x => x.Value - 1);
            var diffdataMA10 = (data / MA10).Select(x => x.Value - 1);

            var diffMA3MA5 = (MA3 / MA5).Select(x => x.Value - 1);
            var diffMA3MA10 = (MA3 / MA10).Select(x => x.Value - 1);
            var diffMA5MA10 = (MA5 / MA10).Select(x => x.Value - 1);

            var SD3 = data.Window(3).Select(x => x.Value.StdDev());
            var SD5 = data.Window(5).Select(x => x.Value.StdDev());
            var SD10 = data.Window(10).Select(x => x.Value.StdDev());

			var RSI3 = data.Window(3).Select(x => RSI_func(x, 3));
			var RSI5 = data.Window(5).Select(x => RSI_func(x, 5));
			var RSI10 = data.Window(10).Select(x => RSI_func(x, 10));

			var K3 = data.Window(3).Select(x => K_func(x, 3));
			var K5 = data.Window(5).Select(x => K_func(x, 5));
			var K10 = data.Window(10).Select(x => K_func(x, 10));

            string Returncolumn = "Return" + Convert.ToString(shiftn);
            string Powercolumn = "Power" + Convert.ToString(shiftn);

            string MA3column = "MA3" + Convert.ToString(shiftn);
            string MA5column = "MA5" + Convert.ToString(shiftn);
            string MA10column = "MA10" + Convert.ToString(shiftn);

            string SD3column = "SD3" + Convert.ToString(shiftn);
			string SD5column = "SD5" + Convert.ToString(shiftn);
			string SD10column = "SD10" + Convert.ToString(shiftn);

            string CompareDataColumn1 = "dataMA3" + Convert.ToString(shiftn);
            string CompareDataColumn2 = "dataMA5" + Convert.ToString(shiftn);
            string CompareDataColumn3 = "dataMA10" + Convert.ToString(shiftn);

            string Compare1column = "MA3MA5" +Convert.ToString(shiftn);
			string Compare2column = "MA3MA10" + Convert.ToString(shiftn);
			string Compare3column = "MA5MA10" + Convert.ToString(shiftn);

            string RSI3column = "RSI3" + Convert.ToString(shiftn);
            string RSI5column = "RSI5" + Convert.ToString(shiftn);
            string RSI10column = "RSI10" + Convert.ToString(shiftn);

            string K3column = "K3" + Convert.ToString(shiftn);
            string K5column = "K5" + Convert.ToString(shiftn);
            string K10column = "K10" + Convert.ToString(shiftn);

            var Features = new FrameBuilder.Columns<DateTime, string>{
                { Returncolumn, data},
                { Powercolumn, data2 },
                { MA3column  , MA3  },
                { MA5column  , MA5  },
                { MA10column , MA10 },
                {Compare1column, diffMA3MA5},
                {Compare2column,diffMA3MA10},
                {Compare3column,diffMA5MA10},
                {CompareDataColumn1, diffdataMA3},
                {CompareDataColumn2, diffdataMA5},
                {CompareDataColumn3, diffdataMA10},
                {SD3column,SD3},
                {SD5column,SD5},
                {SD10column,SD10},
                {RSI3column, RSI3},
                {RSI5column, RSI5},
                {RSI10column, RSI10}, 
                {K3column,K3},
                {K5column,K5},
                {K10column,K10}

            }.Frame;

          
            return Features;
         
		}

        public static double[] Features_engineering_pred(Series<DateTime, double> input)
        {   

            // Features engineering function for generating prediction input

            int shiftn = 0;

			var data = input.Shift(shiftn);

			var data2 = data.Select(x => Math.Pow(x.Value, 2));

			var MA3 = data.Window(3).Select(x => x.Value.Mean());
			var MA5 = data.Window(5).Select(x => x.Value.Mean());
			var MA10 = data.Window(10).Select(x => x.Value.Mean());

			var diffdataMA3 = (data / MA3).Select(x => x.Value -1 );
			var diffdataMA5 = (data / MA5).Select(x => x.Value - 1);
            var diffdataMA10 = (data / MA10).Select(x => x.Value - 1);

            var diffMA3MA5 = (MA3 / MA5).Select(x => x.Value - 1);
            var diffMA3MA10 = (MA3 / MA10).Select(x => x.Value - 1);
            var diffMA5MA10 = (MA5 / MA10).Select(x => x.Value - 1);

            var SD3 = data.Window(3).Select(x => x.Value.StdDev());
			var SD5 = data.Window(5).Select(x => x.Value.StdDev());
			var SD10 = data.Window(10).Select(x => x.Value.StdDev());

			var RSI3 = data.Window(3).Select(x => RSI_func(x, 3));
			var RSI5 = data.Window(5).Select(x => RSI_func(x, 5));
			var RSI10 = data.Window(10).Select(x => RSI_func(x, 10));

			var K3 = data.Window(3).Select(x => K_func(x, 3));
			var K5 = data.Window(5).Select(x => K_func(x, 5));
			var K10 = data.Window(10).Select(x => K_func(x, 10));

			string Returncolumn = "Return" + Convert.ToString(shiftn);
			string Powercolumn = "Power" + Convert.ToString(shiftn);

			string MA3column = "MA3" + Convert.ToString(shiftn);
			string MA5column = "MA5" + Convert.ToString(shiftn);
			string MA10column = "MA10" + Convert.ToString(shiftn);

			string SD3column = "SD3" + Convert.ToString(shiftn);
			string SD5column = "SD5" + Convert.ToString(shiftn);
			string SD10column = "SD10" + Convert.ToString(shiftn);

			string CompareDataColumn1 = "dataMA3" + Convert.ToString(shiftn);
			string CompareDataColumn2 = "dataMA5" + Convert.ToString(shiftn);
			string CompareDataColumn3 = "dataMA10" + Convert.ToString(shiftn);

			string Compare1column = "MA3MA5" + Convert.ToString(shiftn);
			string Compare2column = "MA3MA10" + Convert.ToString(shiftn);
			string Compare3column = "MA5MA10" + Convert.ToString(shiftn);

			string RSI3column = "RSI3" + Convert.ToString(shiftn);
			string RSI5column = "RSI5" + Convert.ToString(shiftn);
			string RSI10column = "RSI10" + Convert.ToString(shiftn);

			string K3column = "K3" + Convert.ToString(shiftn);
			string K5column = "K5" + Convert.ToString(shiftn);
			string K10column = "K10" + Convert.ToString(shiftn);

			var Features = new FrameBuilder.Columns<DateTime, string>{
				{ Returncolumn, data},
				{ Powercolumn, data2 },
				{ MA3column  , MA3  },
				{ MA5column  , MA5  },
				{ MA10column , MA10 },
				{Compare1column, diffMA3MA5},
				{Compare2column,diffMA3MA10},
				{Compare3column,diffMA5MA10},
				{CompareDataColumn1, diffdataMA3},
				{CompareDataColumn2, diffdataMA5},
				{CompareDataColumn3, diffdataMA10},
				{SD3column,SD3},
				{SD5column,SD5},
				{SD10column,SD10},
				{RSI3column, RSI3},
				{RSI5column, RSI5},
				{RSI10column, RSI10},
				{K3column,K3},
				{K5column,K5},
				{K10column,K10}

			}.Frame;

            var row_length = Features.RowCount;
            var col_length = Features.ColumnCount;

            double[] delay_1 = Features.GetRowAt<double>(row_length-1).Values.ToArray<double>(); 
            double[] delay_2 = Features.GetRowAt<double>(row_length-2).Values.ToArray<double>(); 
            double[] delay_3 = Features.GetRowAt<double>(row_length-3).Values.ToArray<double>(); 
            
            double[] features_pred = new double[col_length * 3];

            for (int i = 0; i < col_length; i++)
            {
                features_pred[i] = delay_1[i];
            }
            for (int i = 0; i < col_length; i++)
            {
                features_pred[i+col_length] = delay_2[i];
            }
            for (int i = 0; i < col_length; i++)
            {
                features_pred[i+col_length*2] = delay_3[i];
            }

            return features_pred;
           
        }

        public static double RSI_func(KeyValuePair<DateTime,Series<DateTime,double>> x, int len)
        {   
            // Fuction to caculate the RSI of time-series to measure the momentum

            double positive=0, negative=0;
            int pos_num = 0, neg_num = 0;


            for (int i = 0; i < len; i++)
            {
               var item = x.Value.GetAt(i);

                if(item > 0)
                {
                    positive = positive + item;
                    pos_num++;
                }
                else if(item < 0)
                {
                    negative = negative - item;
                    neg_num++;
                }
            }

			if (pos_num == 0 && neg_num == 0)
			{
				double RS = 0;
				double RSI = 100 - 100 / (1 + RS);
				return RSI;
			}
			else if (pos_num == 0)
			{
				double RS = 0;
				double RSI = 100 - 100 / (1 + RS);
				return RSI;

			}
			else if (neg_num == 0)
			{
				double RS = 100;
				double RSI = 100 - 100 / (1 + RS);
				return RSI;
			}
			else
			{
				double RS = (positive / pos_num) / (negative / neg_num);
				double RSI = 100 - 100 / (1 + RS);
				return RSI;
			}

        }


        public static double K_func(KeyValuePair<DateTime, Series<DateTime, double>> x, int len)
        {   

            // Function to cacualte the K% of time-series to measure the momentum


            double init = 1;

            double[] list = new double[len+1];
            list[0] = init;

            double min_item = 100, max_item = -100;

            for (int i = 0; i < len; i++)
            {   
                
                var item = list[i] * Math.Exp(x.Value.GetAt(i));
                if (item > max_item)
                {
                    max_item = item;
                }

                if (item < min_item)
                {
                    min_item = item;
                }

                list[i+1] = item;
            }

            double K = (list[len] - min_item) / (max_item - min_item);
            return K;
        }


    }



}
