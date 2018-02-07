package neuron;

import java.io.Serializable;

/**
 * @author Lauri Turunen
 *
 * A mapping of neurons used in determining relative positions and weights
 *
 */

public class Map implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 431673095842222751L;
	public int outputsTo;
	public int oc;

	public Map() {
		super();
	}
	
	@Override
	public String toString() {
		
		return "\n This mapping: "+this.outputsTo;
	}

}
