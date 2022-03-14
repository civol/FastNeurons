module FastNeurons

    # Describes a neuron-based reservoir computing the following:
    # R(i+1) = (1-gamma)*R(i) + gamma*f(A*R(i)+W*X(i)+B)
    ## Y = U*R(i+1)
    ## Where R is the vector of the neurons' values, X is the input vector,
    ## Y the output vector, A, W and U are fixed matrices of weights,
    ## B is the fixed vector of biases, and gamma is the decay rate.
    # Where R is the vector of the neurons' values, X is the input vector,
    # A and W are fixed matrices of weights, B is the fixed vector of biases,
    # and gamma is the decay rate.
    class Reservoir

        #### Initialization and configuration of the reservoir. ####

        # Create a new reservoir. <br>
        # @param [n_x] the size of the input.
        # @param [n_r] the number of neurons.
        # @param [act] the activation function.
        ## @param [n_y] the size of the output.
        def initialize(n_x,n_r, act = :sin) #,n_y)
            # Check and sets the size arguments.
            @n_x = n_x.to_i
            @n_r = n_r.to_i
            # @n_y = n_y.to_i

            # Create the activation function: by default hyperbolic tangent.
            @activation = FastNeurons.method(act)

            # Create the distribution for generating parameters: by default
            # random bell.
            @distribution = RandomBell.new(mu: 0, sigma: 1, range: -Float::INFINITY..Float::INFINITY)

            # Sets the decay: by default 0.2
            @gamma = 0.2

            # Create the fixed matrices and vectors.
            self.generate_A
            self.generate_W
            self.generate_B
            # self.generate_U

            # Initialise the neurons.
            self.generate_R
        end

        # Generates a random matrix.
        # @param [w]    the width of the matrix.
        # @param [h]    the height of the matrix.
        # @param [dist] the distribution for generating the random elements.
        def generate_matrix(w,h, dist = @distribution)
            # Generate an array of random elements.
            weights = (w.to_i*h.to_i).times.map { dist.rand }
            # Convert the array to a matrix.
            return Numo::DFloat.asarray(weights).reshape!(w,h)
        end

        # Generates a random vector.
        # @param [w]    the width of the vector.
        # @param [dist] the distribution for generating the random elements.
        def generate_vector(w, dist = @distribution)
            # Generate an array of random elements.
            weights = (w.to_i).times.map { dist.rand }
            # Convert the array to a vector.
            return Numo::DFloat.asarray(weights)
        end

        # Generate the A matrix.
        # @param [dist] the distribution for generating the random elements.
        def generate_A(dist = @distribution)
            @A = generate_matrix(@n_r,@n_r,dist)
        end

        # Generate the W matrix.
        # @param [dist] the distribution for generating the random elements.
        def generate_W(dist = @distribution)
            @W = generate_matrix(@n_r,@n_x,dist)
        end

        # Generate the B vector.
        # @param [dist] the distribution for generating the random elements.
        def generate_B(dist = @distribution)
            @B = generate_vector(@n_r,dist)
        end

        # # Generate the U matrix.
        # # @param [dist] the distribution for generating the random elements.
        # def generate_U(dist = @distribution)
        #     @U = generate_matrix(@n_y,@n_r,dist)
        # end

        # Generate the neurons initial value in R.
        # @param [dist] the distribution for generating the random elements.
        def generate_R(dist = @distribution)
            @R = generate_vector(@n_r,dist)
        end

        # Gets the value of gamma.
        def gamma
            @gamma
        end

        # Sets the value of gamma.
        # @param [val] the new value of gamma.
        def gamma=(val)
            @gamma = val.to_f
        end

        # Freezes the value of the neurons.
        def freeze_R
            @R_frozen = true
        end

        # Unfree the value of the neurnons.
        def unfreeze_R
            @R_frozen = false
        end

        # Gets the value of the neurons.
        def R
            return @R
        end

        # Sets the value of the neurons.
        # @param [nR]  the new value of R.
        def R=(nR)
            if nR.size != @n_r then
                raise "Invalid size for the R vector: #{nR.size}, " +
                      "expecting: #{@n_r}."
            end
            unless nR.is_a?(Numo::DFloat)
                nR = Numo::DFloat.asarray(nR.to_a) unless nR.is_a?(Numo::DFloat)
            end
            @R = nR
        end


        #### Computation of the reservoir. ####

        # Compute one step.
        # @param [vecX] the input vector.
        # @return the resulting vector.
        def step(vecX = @X)
            @X = vecX
            # Compute the new r
            nR = (1.0-@gamma)*@R + 
                 @gamma*@activation.(@A.dot(@R)+@W.dot(vecX)+@B)
            # nR = @activation.(@W.dot(vecX)+@B)
            # Update R if not frozen.
            @R = nR unless @R_frozen
            # Return the computation result.
            return nR
            # # Compute Y.
            # @Y = @U.dot(@R)
            # return @Y
        end
    end
end
