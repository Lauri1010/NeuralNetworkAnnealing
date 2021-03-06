/*
 * Encog Artificial Intelligence Framework v2.x
 * Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2009, Heaton Research Inc., and individual contributors.
 * See the copyright.txt in the distribution for a full listing of 
 * individual contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */

package activationFunction;

import utils.BoundMath;

/**
 * An activation function based on the logarithm function.
 * 
 * This type of activation function can be useful to prevent saturation. A
 * hidden node of a neural network is said to be saturated on a given set of
 * inputs when its output is approximately 1 or -1 "most of the time". If this
 * phenomena occurs during training then the learning of the network can be
 * slowed significantly since the error surface is very at in this instance.
 * 
 * @author jheaton
 * 
 */
public class ActivationLOG implements ActivationFunction {

	/**
	 * The serial id.
	 */
	private static final long serialVersionUID = 7134233791725797522L;

	/**
	 * Implements the activation function. The array is modified according to
	 * the activation function being used. See the class description for more
	 * specific information on this type of activation function.
	 * 
	 * @param d
	 *            The input array to the activation function.
	 */
	public void activationFunction(final double[] d) {

		for (int i = 0; i < d.length; i++) {
			if (d[i] >= 0) {
				d[i] = BoundMath.log(1 + d[i]);
			} else {
				d[i] = -BoundMath.log(1 - d[i]);
			}
		}

	}

	/**
	 * @return The object cloned.
	 */
	@Override
	public Object clone() {
		return new ActivationLOG();
	}

	/**
	 * Create a Persistor for this activation function.
	 * 
	 * @return The persistor.
	 *//*
	@Override
	public Persistor createPersistor() {
		return new ActivationLOGPersistor();
	}*/

	

	/**
	 * @return Return true, log has a derivative.
	 */
	public boolean hasDerivative() {
		return true;
	}

	@Override
	public double activationFunction(double d) {
		
		if (d >= 0) {
			d = BoundMath.log(1 + d);
		} else {
			d = -BoundMath.log(1 - d);
		}
		
		return d;
	}

	@Override
	public double derivativeFunction(double d) {
	
		if (d >= 0) {
			d = 1 / (1 + d);
		} else {
			d = 1 / (1 - d);
		}
		
		return d;
	}


}
