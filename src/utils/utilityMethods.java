package utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;
import neuron.Neuron;
import simulatedAnnealing.NeuralSimulatedAnnealing;
import supervisedLearning.BackPropagationNetwork;

public class utilityMethods {
	
	int cycles=11;
	double heat=40;
	NeuralSimulatedAnnealing anneal =new NeuralSimulatedAnnealing();
    boolean exit=false;
	public NumberFormat pFormat = NumberFormat.getPercentInstance();
	
	public utilityMethods() {
		this.pFormat.setMinimumFractionDigits(1);
	}

	public void saveNeuralNetwork(BackPropagationNetwork feedForward1,String fileName) throws IOException {
		SerializeObject.save(fileName, feedForward1.getNeurons());
	}
	
	@SuppressWarnings("unchecked")
	public BackPropagationNetwork loadNeuralNetwork(BackPropagationNetwork feedForward,String fileName) throws Exception {
		ArrayList<Neuron> n=(ArrayList<Neuron>) SerializeObject.load(fileName);
		feedForward.setNeurons(n);
		return feedForward;
	}
	
	public void simulatedAnnealing(double heat,int cycles,BackPropagationNetwork feedForward) throws Exception{
		this.anneal.setNetworkAndCopyBestSet(feedForward);
		this.anneal.setCycles(cycles);
		this.anneal.setStartTemp(heat);
		anneal.iterate();
	}

	    TimerTask task = new TimerTask()
	    {
	        public void run()
	        {
	        	if(!exit){
	        		return;
	        	}else{
	        		System.exit(0);
	        	}
	        }    
	    };

	    public void exitInput() throws Exception
	    {
	        Timer timer = new Timer();
	        timer.schedule(task, 30*1000 );
	        
	        System.out.println( "Do you wish to exit training? Type yes to exit, you have 30 seconds. Data has been saved in this iteration" );
	        BufferedReader in = new BufferedReader(
	        new InputStreamReader( System.in ) );
	        try {
				String input = in.readLine();
		        if(input.equalsIgnoreCase("yes")){
		        	System.exit(0);
		        }
			} catch (IOException e) {
				e.printStackTrace();
			}
	        
	        timer.cancel();

	    }

	
}
