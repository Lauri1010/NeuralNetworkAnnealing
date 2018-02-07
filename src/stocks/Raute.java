package stocks;

import java.io.IOException;
import java.io.File;
import java.util.Iterator;
import org.apache.poi.EncryptedDocumentException;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.DataFormatter;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;
import utils.utilityMethods;
import supervisedLearning.BackPropagationNetwork;
import activationFunction.ActivationFunction;
import activationFunction.ActivationSigmoid;


public class Raute {
	
	  // NOTE  current prediction input size 10, output size 1
	  private static final int TRAINING_SIZE = 125;
	  /**
	   * The number of input neurons
	   * NOTE when predicting with real data this has to be the same as the original size
	   * (same apples to number of hidden neurons)
	   */
	  private static final int INPUT_SIZE = 10;
	  private static final int OUTPUT_SIZE = 1;
	
	  private double[][] input = new double[TRAINING_SIZE][INPUT_SIZE];
	  private double[][] ideal = new double[TRAINING_SIZE][OUTPUT_SIZE];
	  private utilityMethods utils=new utilityMethods();
	  
	  public double maxError=0;

	   public void runNetwork(String excelFilePath,int train) throws Exception{
			 	
				this.readStockData(excelFilePath,TRAINING_SIZE);
				int[] hidden = new int[1];
				hidden[0]=INPUT_SIZE*45;
				ActivationFunction acFunct=new ActivationSigmoid();
				// double lRate=0.00000768;
				// double lRate=0.005;
				// double lRate=0.000000054;
				double lRate=0.00000768;
				double momentum=0.00000000001;
				double error=1;
				double ep;
				double lastError = Double.MAX_VALUE;
				int eec=0;
				int iec=0;
				int maxEec=2500;
				int maxIec=2500;
				int ma=500;
				int mai=0;
				int i=0;
				int in=0;
				int obc=0;
				int iterations=1000;
				double rheats=0.3;
				double rheatst=0.1;
				int maxIterations=200000;
				double heat=2;
				int cycles=10;
				BackPropagationNetwork feedForward=
				new BackPropagationNetwork(INPUT_SIZE,hidden,OUTPUT_SIZE,acFunct,lRate,momentum,this.input,this.ideal,true,this.maxError);
				// utils.simulatedAnnealing(5, 10, feedForward);

				
				if(train==1){
				
					while(in<iterations){
						i++;
						in++;
						error=feedForward.iteration(false);
						System.out.println("Iteration: "+i+" total deviation: "+error+" ");
						// System.out.println(error);
						if(in == iterations){
							utils.saveNeuralNetwork(feedForward, "raute.net");
							if((error<0.082 && error>-0.00000001)){
								feedForward.run(true);
								System.exit(0);
							}
							if(error>lastError || error==lastError){
								utils.simulatedAnnealing(200,150,feedForward);
							}
							iterations*=2;
						}else if((error<0.082 && error>-0.00000001)|| in>maxIterations){
							System.out.println("Good error level achieved, saving network");
							feedForward.run(true);
							utils.saveNeuralNetwork(feedForward, "raute.net");
							System.exit(0);
							
						}
						
						if(error>lastError){
							eec++;
							feedForward.decreaseLearningRate();
							feedForward.decreaseMomentum();
							if(mai==ma){
								utils.simulatedAnnealing(100,150,feedForward);
								mai=0;
							}
							mai++;
							if(eec>maxEec){
								utils.simulatedAnnealing(100,150,feedForward);
								eec=0;
							}
						}else if(error<lastError){
							eec=0;
							iec=0;
							mai=0;
							feedForward.increaseLearningRate3(feedForward.getLRate()*25);
							feedForward.increaseMomentum(feedForward.getMomentum()*2.5);
						}else if(error==lastError){
							mai++;
							feedForward.increaseLearningRate3(feedForward.getLRate()*25);

							if(mai==ma){
								utils.simulatedAnnealing(150,150,feedForward);
								mai=0;
							}
			
							iec++;
							if(iec>maxIec){
								iec=0;
								utils.simulatedAnnealing(100,150,feedForward);
							}
	
						} 
						if(error>1){
							obc++;
							if(obc>100){
								obc=0;
								utils.simulatedAnnealing(100,150,feedForward);
							}

						}
						lastError=error;
						feedForward.decreaseLearningRate2();
					}
				
				}else{
					
					try {
						feedForward=utils.loadNeuralNetwork(feedForward, "raute.net");
						feedForward.run(true);
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
				}
				
	   }

	
		
		public void readStockData(String excelFilePath,int trainingSize) throws IOException, EncryptedDocumentException, InvalidFormatException{
	
			// Creating a Workbook from an Excel file (.xls or .xlsx)
	        Workbook workbook = WorkbookFactory.create(new File(excelFilePath));
	
	        // Retrieving the number of sheets in the Workbook
	        System.out.println("Workbook has " + workbook.getNumberOfSheets() + " Sheets : ");
	
	        // 1. You can obtain a sheetIterator and iterate over it
	        Iterator<Sheet> sheetIterator = workbook.sheetIterator();
	        System.out.println("Retrieving Sheets using Iterator");
	        while (sheetIterator.hasNext()) {
	            Sheet sheet = sheetIterator.next();
	            System.out.println("=> " + sheet.getSheetName());
	        }
	
	        // Getting the Sheet at index zero
	        Sheet sheet = workbook.getSheetAt(0);
	
	        // Create a DataFormatter to format and get each cell's value as String
	        DataFormatter dataFormatter = new DataFormatter();
/*	        double ideal;
	        double[] input={};
	        boolean cont=true;
	        int rowSampleIndex=1;
	        int i2=1;
        	int previousRi=0;
        	int icount=3;
	        
	        while(rowSampleIndex<trainingSize){

	        	do{
	        		Row row=sheet.getRow(i2);
	        		int ri=previousRi;
	        		int celln=1;
	        		int cutoff=10;
	        		if(ri>0){cutoff=18;};
	        		while(ri<cutoff){
	        			this.input[rowSampleIndex-1][ri]=row.getCell(celln).getNumericCellValue();
	        			ri++;
	        			celln++;
	        		}
	        		previousRi=ri;
	        		i2++;
	        		if(i2 % 3 == 0){
	        			double[] ia={row.getCell(6).getNumericCellValue()};
	        			this.ideal[rowSampleIndex-1]=ia;
	        		};
	        	}while(i2<=icount);
	        	rowSampleIndex++;
	        	icount=rowSampleIndex*2;
	        	previousRi=0;

	        }*/

	        for (Row row: sheet) {
	        	int rn=row.getRowNum();
	        	if(rn<=trainingSize){
		        	double Bid=row.getCell(1).getNumericCellValue();
		        	double Ask=row.getCell(2).getNumericCellValue();
		        	double openingPrice=row.getCell(3).getNumericCellValue();
		        	double highPrice=row.getCell(4).getNumericCellValue();
		        	double lowPrice=row.getCell(5).getNumericCellValue();
		        	double closingPrice=row.getCell(6).getNumericCellValue();
		        	double averagePrice=row.getCell(7).getNumericCellValue();
		        	double totalVolume=row.getCell(8).getNumericCellValue();
		        	double turnover=row.getCell(9).getNumericCellValue();
		        	double trades=row.getCell(10).getNumericCellValue();
		        	if(rn<trainingSize){
			        	double[] input={Bid,Ask,openingPrice,highPrice,lowPrice,closingPrice,averagePrice,totalVolume,turnover,trades};
		        		// double[] input={Ask,openingPrice,closingPrice,averagePrice,totalVolume,turnover,trades};
		        		// double[] input={Bid,Ask,openingPrice,closingPrice,turnover,trades};
		        		// double[] input={Bid,Ask,closingPrice,turnover,trades};
		        		// double[] input={closingPrice,turnover,trades};
			        	this.input[rn]=input;
		        	}
		        	if(rn>0){
		        		int ii=rn-1;
		        		this.maxError+=closingPrice;
		        		double[] ia={closingPrice};
		        		this.ideal[ii]=ia;
		        	}
	        	}
	        }
	        // Closing the workbook
	        workbook.close();
			
		}
		
		
}

