using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Deedle;


namespace BacktestSystem
{
    public class Backtest
    {
        public double Net_value = new double();

        public Series<int, string> namelist = Frame.ReadCsv("data/Mapping_Table.csv")
            .GetColumn<string>("Best Tracking ETF");
        public List<Series<DateTime, double>> Hisc_data = new List<Series<DateTime, double>>();
        public List<string> ETF_holding = new List<string>();
        public List<double> ETF_bought_price = new List<double>();
        public List<double> Allocation = new List<double>();

        public void Init()
        {   
            // Set the initial net value equals to 1 
            Net_value = 1;

            List<string> NameListArrary = new List<string>();

            var Data = Frame.ReadCsv("data/ETF Returns.csv").IndexRows<DateTime>("Date");

            for (int i = 0; i < namelist.KeyCount; i++)
            {
                var data = Data.Rows.Select(x => x.Value.GetAs<double>(namelist[i]));
                NameListArrary.Add(namelist[i]);
                Hisc_data.Add(data);
            }

            Hisc_data = Adjust_etf_fx(Hisc_data, NameListArrary.ToArray());

        }

        public double Rebalance(DateTime date, string[] ETFs, double[] allocation)
        {
            // Update Net value 
            if (ETF_holding.Count==0)
            {
                for (int i = 0; i < ETFs.GetLength(0); i++)
                {
                    ETF_holding.Add(ETFs[i]);
                    Allocation.Add(allocation[i]);

                    for (int j = 0; j < namelist.KeyCount; j++)
                    {
                        if (namelist[j] == ETFs[i])
                        {
                            var Price = Hisc_data[j].Get(date);
                            ETF_bought_price.Add(Price);
                        }
                        else
                        {
                            continue;
                        }
                    }

                }
            }
            else
            {
                // Caculate the new net value 

                List<double> Current_Price = new List<double>();

                for (int i = 0; i < ETF_holding.Count; i++)
                { 
                    for (int j = 0; j < namelist.KeyCount; j++)
                    {
                        if (namelist[j] == ETF_holding[i])
                        {
                            var Price = Hisc_data[j].Get(date);
                            Current_Price.Add(Price);
                        }
                        else
                        {
                            continue;
                        }
                    }
                }

                double OverallReturn = 0 ;
                for (int t = 0; t < Current_Price.Count; t++)
                {
                    double returns= Current_Price[t] / ETF_bought_price[t]-1;
                    OverallReturn += Allocation[t] * returns;
                }

                Net_value = Net_value* (1+OverallReturn);

                // Reset the information of ETF holding, bought price and allocation 

                ETF_bought_price = new List<double>();
                ETF_holding = new List<string>();
                Allocation = new List<double>();

                for (int i = 0; i < ETFs.GetLength(0); i++)
                {
                    ETF_holding.Add(ETFs[i]);
                    Allocation.Add(allocation[i]);
                    for (int j = 0; j < namelist.KeyCount; j++)
                    {
                        if (namelist[j] == ETFs[i])
                        {
                            var Price = Hisc_data[j].Get(date);
                            ETF_bought_price.Add(Price);
                        }
                        else
                        {
                            continue;
                        }
                    }
                   
                }
            }

            Console.WriteLine("Current Net Value is {0}", Net_value);
            return Net_value;
            
        }

        public static Series<DateTime, double> Price2Return(Series<DateTime, double> data, string Return_type = "log")
        {

            if (Return_type == "log")
            {
                return (data / data.Shift(1)).Log();
            }

            var relative_return = (data / data.Shift(1)).Diff(1);
            return relative_return;
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
                    List_data.Add(data[i] / GBPUSD.Between(FirstKey, LastKey));
                }
                else if (CRNCY == "EUR")
                {
                    var FirstKey = data[i].FirstKey();
                    var LastKey = data[i].LastKey();
                    List_data.Add(data[i] / GBPEUR.Between(FirstKey, LastKey));
                }
                else
                {
                    List_data.Add(data[i]);
                }

            }

            return List_data;
        }
    }
}
