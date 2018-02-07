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
 * An activation function based on the sin function.
 * 
 * @author jheaton
 */
public class ActivationSigmoid2 extends BasicActivationFunction {

	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = 5622349801036468572L;

	/**
	 * @return The object cloned;
	 */
	@Override
	public Object clone() {
		return new ActivationSigmoid();
	}
	
	/**
	 * @return Return true, sigmoid has a derivative.
	 */
	public boolean hasDerivative() {
		return true;
	}

	@Override
	public double activationFunction(double d) {
       double e = BoundMath.exp(d);
       e=e*9;
       double y = Math.round(- 1.0/(1.0+e));
       return y;
	   
	}

	@Override
	public double derivativeFunction(double d) {
        double e = Math.round(BoundMath.exp(d));
        double y = Math.round(- e/((1.0+e)*(1.0+e)));
        return y;

	}
}
