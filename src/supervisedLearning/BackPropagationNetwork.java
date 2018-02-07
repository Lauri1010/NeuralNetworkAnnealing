package supervisedLearning;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;
import neuron.Connection;
import neuron.Map;
import neuron.NeuralMapping;
import neuron.Neuron;
import activationFunction.ActivationFunction;
import utils.BoundNumbers;
import utils.ObjectCloner;

/**
 * @author Lauri Turunen 
 * <br/><br/>
 *
 */

public class BackPropagationNetwork
{

	private int outputLayer=3;
	private int j=0; 
	/**
	 * Used to call neural mapping class methods.  
	 */
	private NeuralMapping nm;
	/**
	 * The number of input neurons in the network.
	 */
	private int thisInputs=0;
	/**
	 * The number of hidden neurons in the network (an array because there may be more than one).
	 */
	private int thisHidden[]=new int[2];
	/**
	 * The number of output neurons in the neural network. 
	 */
	private int thisOutputs=0;
	/**
	 * The data location used to reference for input and ideal data. 
	 */
	public int dataLocation=0;
	/**
	 * The 
	 * neuron vector used to store the different neurons
	 */
	protected ArrayList<Neuron> neurons;
	/**
	 * The total error (RMS) for the entire training set. 
	 */
	private double totalError=0.0;
	private double maxError=0;
	/**
	 * The best error rate thus far 5000000.0 for comparison purposes. 
	 */
	private double bestErrorRate=5000000.0;
	/**
	 * A best set vector that can be used to store a best result of the training. 
	 */
	Vector<Neuron> bestSet; 
	/**
	 * the starting point for layer 1 neurons.
	 */
	int i1;
	/**
	 * The cutoff point for layer 1 neurons. 
	 */
	int t1;
	/**
	 * The starting index for layer 2 neurons
	 */
	int i2;
	/**
	 * The cutoff point for layer 2 neurons. 
	 */
	int t2;
	/**
	 * The starting point for layer 3 neurons
	 */
	int i3;
	/**
	 * The cutoff point for layer 3 neurons. 
	 */
	int t3;
	/**
	 * The number of epochs (used with iterate method). 
	 */
	private int epoch;
	/**
	 * A variable representing the current iteration.
	 */
	int p=0;
	/**
	 * Stores the weight sums for the upper layer. Used for backpropagation.
	 * References the lower level weight to the weight of the upper layer neuron connection. 
	 */
	private HashMap<Integer, Double> weightSum;
	/**
	 * Stores the upper neuron connection  delta value. 
	 */
	private HashMap<Integer, Double> upperDelta;
	/**
	 * Stores the error data for output layer neuron(s).
	 */
	private HashMap<Integer, Double> errorData;
	/**
	 * Holds the data of the actual prediction data (when predicting with the real test data). 
	 */
	private double[] actualDataReturn;
	/**
	 * The current error for the iteration (actual - desired). 
	 */
	private double currentError;
	/**
	 * The activation function variable (interface class instance). 
	 */
	protected ActivationFunction acFunct;
	/**
	 * The desired result data (in training). 
	 */
	protected double[][] ideal;
	/**
	 * The given input data. 
	 */
	protected double[][] input;
	/**
	 * The actual data input for the testing phase. 
	 */
	protected double[] actualDataInput;
	/**
	 * The learning rate (alpha)
	 */
	protected double lRate;
	/**
	 * The momentum; how much past weights affect the weigh change calculation).
	 */
	protected double momentum;
	/**
	 * The output layer size (used in predicting with actual data). 
	 */
	protected int outputSize;
	
	NumberFormat f;
	
	double traningValueSum;
	
	double bPercentage;
	
	/**
	 * The error percentage
	 */
	double peError;
	
	/**
	 * The default momentum
	 */
	double dMomentum;
	
	/**
	 * The default learning rate
	 */
	double dLearningRate;
	
	double lRdescent;
	
	double mDescent;
	
	double firstValue;
	
	boolean fError;
	
	NumberFormat pFormat = NumberFormat.getPercentInstance();
	
	int listSize=0;
	
	/**
	 * Creates a Backpropagation network using the parameters.
	 * 
	 * @param inputs  the number of input neurons
	 * @param hidden the number of hidden neurons
	 * @param output the number of output neurons
	 * @param acFunction the activation function. 
	 * @param learningRate the learning rate
	 * @param momentum the momentum (how much past weight affects the training)
	 * @param input the input data
	 * @param ideal the ideal data
	 * @param train determines whether the system is on full trainig mode
	 */
	public BackPropagationNetwork(int inputs, int hidden[], int output,ActivationFunction acFunction,
			double learningRate, double momentum, final double[][] input, final double[][] ideal,boolean train,double maxError)
	{
		this.pFormat.setMinimumFractionDigits(1);
		this.maxError=maxError;
		thisInputs=inputs;
		thisHidden=hidden;
		thisOutputs=output;
		listSize=inputs+hidden[0]+output;
		this.acFunct=acFunction;
		this.momentum=momentum;
		this.input=input;
		this.ideal=ideal;
		this.lRate=learningRate;
		weightSum=new HashMap<Integer,Double>();
		upperDelta=new HashMap<Integer, Double>();
		errorData=new HashMap<Integer, Double>();
		neurons = new ArrayList<Neuron>(listSize);
		bestSet=new Vector<Neuron>(listSize);
		epoch=0;
		this.f= NumberFormat.getPercentInstance();  
		this.f.setMinimumFractionDigits(1);  
		this.f.setMaximumFractionDigits(1); 
		this.dMomentum=momentum;
		this.dLearningRate=learningRate;
		this.lRdescent=5.0;
		this.mDescent=1.0;
		this.fError=true;
		
		i1=0;
		t1=thisInputs;
		i2=thisInputs;
		t2=thisHidden[0]+thisInputs;
		i3=thisHidden[0]+thisInputs;
		t3=thisHidden[0]+thisInputs+thisOutputs;
		
		for(int i=0;i<inputs;i++)
		{
			neurons.add(new Neuron(1));
		}
		for(int j=0;j<hidden[0];j++)
		{
			neurons.add(new Neuron(2));
		}
		for(int k=0;k<output;k++)
		{
			neurons.add(new Neuron(3));
		}
		
		this.nm=new NeuralMapping();
		createNetwork();
		if(train){calclulateTraningValueSum();}
	}
	/**
	 * Creates a neural network using the data of number of neurons in each layer. 
	 */
	public void createNetwork(){
		neurons=nm.feedForwardSetNeuralNet(neurons,thisInputs,thisHidden,thisOutputs);		
		neurons=nm.mapOututs(neurons, thisInputs, thisHidden, thisOutputs);
		System.out.println("\n Network created \n");
	}
	/**
	 * Iterates the network. there are three phases: <br/><br/>
	 * Phase A: <br/>
	 * 1. input patterns<br/>
	 * 2. feedforward: Pass the values forward (with defined weights)<br/>
	 * 3. backpropagate: propagate backwards and calclulate the error sum for each connection.<br/>
	 * 4. Manage results: increment total error <br/><br/>
	 * Phase B: <br/>
	 * 1. Learn: adjust weights with the error sum <br/>
	 * 2. Store best result and print the current total error. 
	 * @throws Exception 
	 */
	public void iterate() throws Exception
	{
		do
		{
		  this.p++;
			
			do
			{
				inputPatterns();
				feedForward(false);
				backPropagate();
				learn();
				manageResults();
				System.out.println("" +
						"the desired value "+ideal[this.dataLocation][0]+"  The current output "
						+Math.sqrt(this.neurons.get(this.neurons.size()-1).getActivationOutput())+" Error percentage "+this.getErrorPercentage());
				dataLocation++;
			}
			while(dataLocation<input.length);
			
			dataLocation=0;
			calculateErrorPercentage();
			printResult();
			testBestSet();
			errorData.clear();
			this.peError=0.0;
		}
		while(this.p<epoch);

	}
	/**
	 * Does one iteration of the network. there are three phases: <br/><br/>
	 * NOTE: only works for neural networks with a single output (can be modified)
	 * Phase A: <br/>
	 * 1. input patterns<br/>
	 * 2. feedforward: Pass the values forward (with defined weights)<br/>
	 * 3. backpropagate: propagate backwards and calclulate the error sum for each connection.<br/>
	 * 4. Manage results: increment total error <br/><br/>
	 * Phase B: <br/>
	 * 1. Learn: adjust weights with the error sum <br/>
	 * 2. Store best result and return the current error. 
	 * @throws Exception 
	 */
	public double iteration(boolean details) throws Exception
	{
		double totalError=0;
		do
		{
			inputPatterns();
			feedForward(false);
			backPropagate();
			manageResults();
			double desired=ideal[this.dataLocation][0];
			double actual=Math.sqrt(this.neurons.get(this.neurons.size()-1).getActivationOutput());
			if(Double.isNaN(actual)){actual=0;};
			double diff=desired-actual;
			if(Double.isNaN(diff)){diff=desired;};
			if(diff<0){diff= diff*-1;};
			totalError+=diff;
			if(details){System.out.println("the desired value "+desired+"  The current output "+actual+" difference: "+diff +" Current total error: "+totalError);};
			dataLocation++;
		}
		while(dataLocation<input.length);
		dataLocation=0;
		
		this.totalError=totalError;
		this.peError=totalError/maxError;
		double error=this.peError;
		testBestSet();
		learn();
		errorData.clear();
		this.peError=0.0;
		return error;
	}

	/**
	 * A method that can be used only for the feedfoward. Useful for other methods than backpropagation. 
	 * 
	 * @param details
	 * 		whether to show the errors with each training case
	 */
	
	public void run(boolean details)
	{
	
		do
		{
			inputPatterns();
			feedForward(false);
			manageResults();
			double desired=ideal[this.dataLocation][0];
			double actual=Math.sqrt(this.neurons.get(this.neurons.size()-1).getActivationOutput());
			if(Double.isNaN(actual)){actual=0;};
			double diff=desired-actual;
			if(Double.isNaN(diff)){diff=desired;};
			if(diff<0){diff= diff*-1;};
			totalError+=diff;
			if(details){
				// System.out.println("the desired value "+desired+"  The current output "+actual+" difference: "+diff +" Current total error: "+totalError);
				System.out.print(desired);
				System.out.print(";");
				System.out.print(actual);
				System.out.println("");
			};
			this.dataLocation++;
		}
		while(this.dataLocation<input.length);
		
		this.dataLocation=0;
	}
	
	/**
	 * Used to input patterns for the input layer neurons
	 */
	public void inputPatterns() {
		
		int a=0;
		int i=-1;
		
		for(Neuron n: neurons)
		{
			i++;
			if(n.getLayer()==1)
			{
				n.getC().setInput(BoundNumbers.bound(input[dataLocation][a]));
				a++;
				if(a%thisInputs==0)
				{	
					a=0;
				}
				neurons.set(i, n);
			}

		}
	}

	/**
	 * Backpropagates thrugh the network and calculates the adjustments of weights.
	 * <br/><br/>
	 * For each output layer neuron connection a call is made to setOutputError(c,n).
	 * For each Hidden layer neuron a call is made to setHiddenError(c,n). 
	 * 
	 */
	public void backPropagate() {

		int i=neurons.size()-1;
		int co=thisInputs-1;
		Neuron ln=this.neurons.get(this.neurons.size()-1);
		for(Connection c:ln.getCList2()){
			if(ln.getLayer()==3){
				setOutputError(c,ln);
			}
		}
		while(i>co){
			i--;
			Neuron n=neurons.get(i);
			for(Connection c:n.getCList2()){
				if(n.getLayer()==2){
					setHiddenError(c, n);
				}
			}
		}

	}
	/**
	 * @param n
	 * 	The output neuron used to calculate the error.
	 *  Note not customized for more than one output neuron.  
	 */
	public void setError(Neuron n) {
		double error=n.activationOutput-ideal[this.dataLocation][0];  
		this.currentError=Math.pow(error, 2);
		errorData.put(n.id,error);
	}

	
	/**
	 * Sets the output layer error and stores a reference to upperDelta Hashtable for further calculations
	 * Sums the error data for the output layer connections.<br/><br/>
	 * The calclulation: <br/>
	 * 
	 * Delta: error * Activation of summed input <br/><br/>
	 * Error sum: delta * activation output of the previous layer neuron (on feedforward). <br/>
	 * 
	 * <br/>
	 * @param c the current connection
	 * @param n the current neuron object
	 */
	public void setOutputError(Connection c, Neuron n) { 
/*		double ed=errorData.get(c.thisNeuron);
		double delta=ed*acFunct.derivativeFunction(n.input);
		double ai=neurons.get(c.from).getActivationOutput();
		double sumError=delta*ai;
		c.sumError(sumError);
		upperDelta.put(c.from, delta);*/
		double delta=acFunct.derivativeFunction(n.input);
		double ed=errorData.get(c.thisNeuron);
		c.sumError(ed*delta*neurons.get(c.from).getActivationOutput());
		upperDelta.put(c.from, ed*delta);

	}
	/**
	 * Sets the hidden layer Delta value combined with the activation output multiplication of the upper layer
	 * <br/><br/>
	 * The calculation:
	 * Delta: sum of all weights to connection between hidden and output neurons * derivate of summed input to neuron<br/><br/>
	 * weight sum: previous layer neurons activation output (in feedforward) * (-delta)
	 * <br/>
	 * @param c  the current connection
	 * @param n  the current neuron object
	 */
	public void setHiddenError(Connection c, Neuron n){
/*		double ud=upperDelta.get(c.thisNeuron);
		double ws=weightSum.get(c.thisNeuron);
		double ad=acFunct.derivativeFunction(n.input);
		double delta=ud*ws*ad;
		delta=Math.abs(delta)*-1;
		double se=neurons.get(c.from).getActivationOutput();
		double sse=se*delta;
		c.sumError(sse); */
		c.sumError((Math.abs(upperDelta.get(c.thisNeuron)*weightSum.get(c.thisNeuron)*acFunct.derivativeFunction(n.input))*-1)*(neurons.get(c.from).getActivationOutput()));
	}
	/**
	 * A method used to adjust the weights of the neural network based on the stored data. After each call the error for each neuron 
	 * is zeroed out.
	 */
	public void learn()
	{
		for(Neuron n:neurons)
		{
			int layer=n.getLayer();
			if(layer==2 || layer==3)
			{
				for(Connection c:n.getCList2())
				{
					c.adjustWeights(lRate,momentum);
					c.clearError();
				}
			}
		}
	}
	
	public void outputNeuronsToConsole()
	{ 
		for(Neuron n:neurons)
		{
			System.out.println(n.toString());
			System.out.println("");
		}
	}
	/**
	 * A method for testing and saving the best error results.
	 */
	public void testBestSet() throws Exception{
		
		if(totalError<bestErrorRate)
		{
			bestErrorRate=totalError;
			this.bPercentage=this.peError;
			this.peError=0.0;
			totalError=0.0;
			currentError=0.0;
		}
		else
		{
			totalError=0.0;
			currentError=0.0;
			this.peError=0.0;
		}
	}
	
	/**
	 * A feedforward method using output maps. Passes the input values forward based on the neuron mappings <br/><br/>
	 * 	Note this method requires the neurons to be in correct order in the vector
	 * @param actual
	 */
	public void feedForward(boolean actual){

		int i=neurons.size()-1;
		int co=thisInputs-1;
		Neuron ln=neurons.get(this.neurons.size()-1);
		for(Connection c:ln.getCList2()){
			if(ln.getLayer()==3){
				ln.processInput(this.acFunct, true);
				weightSum.put(c.from, c.weight);
				if(!actual){setError(ln);}
			}
		}
		while(i>co){
			i--;
			Neuron n=neurons.get(i);
			n.processInput(this.acFunct, false);
			for(Map m:n.outputMap){
					Connection c=neurons.get(m.outputsTo).cList2[m.oc];
					c.input=n.activationOutput;
			}
		}
		
	}
	
	public int runAnnealing(double heat,double cycles,double oError, int i) throws Exception{

		double currentTemp=heat;
		double stoptemp=heat*0.65;
		double pError=oError;
		double peError=oError;
		double bpError=Double.MAX_VALUE;
		final double ratio = Math.exp(Math.log(stoptemp / heat) / (cycles - 1));
		boolean stop=false;
		double stopRate=Double.MAX_VALUE;
		double stopRatet=0.85;
		
		if(oError<0.1){
			stopRatet=0.9999;
		}else if(oError<0.15){
			stopRatet=0.9995;
		}else if(oError<0.25){
			stopRatet=0.995;
		}else if(oError<0.45){
			stopRatet=0.95;
		}else if(oError<0.60){
			stopRatet=0.90;
		}
		
		do{
			int c=0;
			do{
				
				this.randomize(currentTemp,heat);
				do{
					peError=pError;
					pError=this.iteration(false);
					i++;
					System.out.println("Iteration: "+i+" total deviation: "+pError+" ");
					stopRate=pError/oError;
					if(stopRate<stopRatet){
						stop=true;
					}
				}while(pError<peError && !stop);

				c++;

			}while(c<cycles && !stop);
			
			currentTemp *= ratio;
			
		}while(currentTemp>stoptemp && !stop);

		return i;
	}
	
	public void randomize(double currentTemp,double startTemp){
		int i=0;
		int ns=this.neurons.size();
		while(i<ns){
			Neuron n =this.neurons.get(i);
			if(n.getLayer()>1){
				for(Connection c:n.getCList2()){
					double add = 0.5 - (Math.random());
					add /= startTemp;
					add *= currentTemp;
					c.setWeight(c.getWeight()+add);
					c.setpWeight(c.getWeight()+add);
				}
			}
			i++;
		}
		
	}
	
	/**
	 * Used to feed actual prediction data to the system
	 * 
	 * @return
	 */
	public double[] actualDataPrediction()
	{
		int e=0;
		
		do{
			if(dataLocation<this.actualDataInput.length){
				inputSinglePattern(this.actualDataInput);
				feedForward(true);
				actualDataReturn[e]=this.neurons.get(this.neurons.size()-1).getActivationOutput();
			}
			e++;
		}
		while(e<this.outputSize);
		
		return actualDataReturn;
	}
	
	public void inputSinglePattern(double[] pattern){
		dataLocation=0;
		for(Neuron n: neurons){
			if(n.getLayer()==1){
				n.getC().setInput(pattern[dataLocation]);
				dataLocation++;
			}
		}
	}
	
	public void manageResults()
	{
		this.totalError+=Math.round(currentError);
	}
	public void printResult(){
		System.out.println("Iteration: "+this.p+" Error:  "+this.peError+" ");  
	}
	/**
	 * calclulates the error percentage based on the first value (in decimal format). 
	 */
	public void calculateErrorPercentage(){ 
		if(fError){
			this.firstValue=this.totalError;
			fError=false;
			this.peError=1.0;
		}else{
			this.peError=this.totalError/this.firstValue;
		}
	
		
	}
	public void calclulateTraningValueSum(){
		for(int i=0;i<this.ideal.length;i++){
			this.traningValueSum+=this.ideal[i][0];
		}
	}
	
	public void setEpoch(int epoch)
	{
		this.epoch=epoch;
	}
	
	public void printBestResult() throws Exception{
		if(this.bPercentage<0.001){
			System.out.println("Best error result: 0.0 % ");
		}else{
			System.out.println("Best error result: "+f.format(this.bPercentage)+" ");
		}
	}

	public double getTotalError() {
		return totalError;
	}

	public void setTotalError(double totalError) {
		this.totalError = totalError;
	}

	public int getInputLocation() {
		return dataLocation;
	}

	public void setInputLocation(int inputLocation) {
		this.dataLocation = inputLocation;
	}

	public double getBestErrorRate() {
		return bestErrorRate;
	}

	public void setBestErrorRate(double bestErrorRate) {
		this.bestErrorRate = bestErrorRate;
	}
	
	@SuppressWarnings("unchecked")
	public void setNeurons(ArrayList<Neuron> neurons) throws Exception {
		this.neurons = (ArrayList<Neuron>) ObjectCloner.deepCopy(neurons);
	}

	public ArrayList<Neuron> getNeurons() {
		return neurons;
	}

	public double[] getActualDataReturn() {
		return actualDataReturn;
	}

	public void setActualDataReturn(double[] actualDataReturn) {
		this.actualDataReturn = actualDataReturn;
	}
	public double[] getActualDataInput() {
		return actualDataInput;
	}
	public void setActualDataInput(double[] actualDataInput) {
		this.actualDataInput = actualDataInput;
	}
	public int getOutputSize() {
		return outputSize;
	}
	public void setOutputSize(int outputSize) {
		this.outputSize = outputSize;
	}
	public double getPeError() {
		return peError;
	}
	public String getErrorPercentage(){
		return f.format(this.peError);
	}
	
	public void setPeError(double peError) {
		this.peError = peError;
	}

	public double getMomentum() {
		return momentum;
	}
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
	public double getLRate() {
		return lRate;
	}
	public void setLRate(double rate) {
		lRate = rate;
	}
	public void decreaseLearningRate(){
		this.lRate*=0.5;
	}
	public void increaseLearningRate(double rate){
		if(this.lRate<0.1){
		this.lRate*=rate;}
	}
	public void largeMomentumIncrease(){
		this.momentum+=0.001;
	}
	public void increaseSmallLearningRate(){
		this.lRate+=0.0000004;
	}
	
	public void increaseLearningRate2(double max){
		if(this.lRate<max){this.lRate*=1.53981;}
	}
	
	public void increaseLearningRate3(double max){
		if(this.lRate<max){this.lRate*=4;}
	}
	
	public void increaseLearningRateNew(){
		this.lRate*=2.5;
	}
	public void increaseMomentum(double maximum){
		if(this.momentum<maximum){;
		this.momentum*=1.0002;
		}
	}
	public void decreaseMomentum(){
		if(this.momentum>0.0000001){
		this.momentum*=0.00015;}
	}
	public void setDefaultMomentum(){
		this.momentum=dMomentum;
	}
	
	public void setLearningRate(double lRate){
		
		this.lRate=lRate;
		
	}
	
	public void setWhenStoppedRates(){
		this.lRate*=10;
		this.momentum*=1.1;
	}
	
	public void setDefaultLearningRate(){
		this.lRate=this.dLearningRate+0.0001;
	}
	public void setBestNeurons() throws Exception{
		this.neurons=(ArrayList<Neuron>) ObjectCloner.deepCopy(bestSet);
	}
	public void setRandomRates(){
		this.lRate=0.00001 * (double)Math.random() - 0.05;
		this.momentum=0.00001 * (double)Math.random() - 0.05;
	}
	public void decreaseLearningRateGradual(){
		this.lRdescent*=0.9999999999995;
		this.lRate=this.dLearningRate*this.lRdescent;
	}
	
	public void decreaseLearningRate2(){
		this.lRate*=0.9999999999;
	}
	
	
	
}
