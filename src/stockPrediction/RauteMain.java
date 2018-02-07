package stockPrediction;

import stocks.Raute;

/**
 * @author Lauri Turunen
 *	Predicts the S&P500 index using backpropagation. 
 *  Uses methods from a S&P prediction program by Jeff Heaton
 *  
 *  The standard number of neurons: 11, 22, 1
 *  
 */

public class RauteMain {

	
	 // Date Bid	Ask	Opening price	High price	Low price	Closing price	Average price	Total volume	Turnover	Trades
	 public static void main(String[] args) throws Exception{
		 int method=Integer.parseInt(args[0]);
		 String excelFile=System.getProperty("user.dir");
		 String trainFileName="\\RAUTE-train.xlsx";
		 String predictFilePath="\\RAUTE-predict.xlsx";
		 
		 if(method==1){
			 excelFile+=trainFileName;
		 }else if(method==2){
			 excelFile+=predictFilePath;
		 }

		 Raute raute=new Raute();
		 raute.runNetwork(excelFile,method);

	 }
	 
	
	

}
