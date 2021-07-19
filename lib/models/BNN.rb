module FastNeurons
  # Describes a standard fully connected BNN trainer based on backpropagation.
  # @example learning of xor
  #  data = [[0,0],[0,1],[1,0],[1,1]]
  #  t = [[0],[1],[1],[0]]
  #  nn = FastNeurons::BNN.new([2, 2, 1], [:Tanh, :Linear])
  #  nn.randomize
  #  data.each_with_index do |inputs, i|
  #    nn.input(inputs, t[i])
  #    nn.run(1)
  #  end
  # @since 1.0.0
  class BNN < NN

    # Binarize the weights and round the biases.
    def binarize!
        @weights.map! { |w| FastNeurons.sign(w) }
        @biases.map! { |b| b.round }
    end


    # Compute multiply accumulate of inputs, weights and biases.
    # z = weights * inputs + biases
    # @param [Integer] row the number of layer currently computing
    # @since 1.0.0
    def compute_z(row)
      # BLAS.gemm performs the following calculations.
      #   C = (alpha * A * B) + (beta * C)
      # In this case, the calculation results are stored in Matrix C.
      # For this reason, need to duplicate the @biases[row] value in @z[row] in advance.
        # puts "@weights[row].shape=#{@weights[row].shape} @a[row].shape=#{@a[row].shape}"
        @z[row] = FastNeurons.sign(@weights[row]).dot(@a[row]) + @biases[row].round
    end

    # # Compute neurons statuses.
    # # Apply activation function to z.
    # # @param [Integer] row the number of layer currently computing
    # # @since 1.0.0
    # def compute_a(row)
    #   @a[row+1] = @antiderivatives[row].call(@z[row])
    # end

  end
end
