module FastNeuronsCNN

    # Default values.
    
    RATE_DEFAULT = 0.01

    RAND_DEFAULT = proc { Random.rand(-1.0..1.0) }


    ## Helping methods and class

    # Some often used loss functions.
    # Format: [function proc, derivate proc]

    # Mean square calculator.
    MeanSquare = [ proc { |y,t| ((y-t).square).sum / y.size },
                   proc { |y,t| (y-t) * (2.0/y.size) } ]


    # Some often used activation functions.
    # Format [function proc, derivate proc] where the derivate take as
    # argument the input but also the output of the function since for
    # several cases (e.g., sigmoid) the derivate is computed with the output
    # of the function and not the input.

    Sigmoid = [ proc { |z| 1.0/(Numo::DFloat::Math.exp(-z)+1.0) },
                proc { |z,a| (-a + 1.0) * a } ]

    Tanh    = [
        proc do |z|
            pos_exp = Numo::DFloat::Math.exp(z)
            neg_exp = Numo::DFloat::Math.exp(-z)
            (pos_exp - neg_exp) / (pos_exp + neg_exp)
        end,
        proc do |z,a|
            1.0 - (a.square)
        end ]

    ReLU    = [ proc { |z| (z+z.abs) / 2.0 },
                proc { |z,a| z.gt(0.0).cast_to(Numo::DFloat) } ]

    LeakyReLU = [ proc { |z|   z.gt(0.0).cast_to(Numo::DFloat)*z +
                               z.le(0.0).cast_to(Numo::DFloat)*z*0.01 },
                  proc { |z,a| z.gt(0.0).cast_to(Numo::DFloat) + 
                               z.le(0.0).cast_to(Numo::DFloat)*0.01 } ]

    Sign    = [ proc { |z|   z.sign * 2.0 - 1.0 },
                proc { |z,a| 1.0 } ]

    
    # Class for a fast convolution.
    #
    # Also includes a method for describing a convolution layer.
    #
    # @example assuming we want to make a step=1 pad=1 convolution of
    #          shape (128,128) with filer of shape (3x3) of matrix m by
    #          filter g:
    #          convolve = Convolve.new([128,128],[3,3],1,1)
    #          convolve.(m,f)
    # NOTE: the step and pad are not usable yet.
    class Convolve

        ## Build a new convolution algorithm.
        #  @param m_shape the shape of the matrices to covolute
        #  @param f_shape the shape of the filters
        #  @param step the step of convolution
        #  @param pad the padding of convolution.
        def initialize(m_shape,f_shape,step = 1, pad = 0)
            # Get and check the shapes.
            @m_shape = m_shape.to_a[0..1]
            @f_shape = f_shape.to_a[0..1]
            # Compute the shape of the result.
            @r_shape = [@m_shape[0]-(@f_shape[0]/2)*2,
                        @m_shape[1]-(@f_shape[1]/2)*2]
            @r_shape[0] += 1 if @f_shape[0].even?
            @r_shape[1] += 1 if @f_shape[1].even?
            puts "@r_shape=#{@r_shape}"
            # Get and check the step and the padding.
            @step = step.to_i
            @pad = pad.to_i
            # Generate the matrix of indices for picking the elements for the
            # convolution.
            # Compute the indices for the first convolution product.
            first_indices = Numo::DFloat.new(@f_shape[1]).seq
            (@f_shape[0]-1).times do |i|
                line = Numo::DFloat.new(@f_shape[1]).seq+(i+1)*@m_shape[1]
                first_indices = first_indices.concatenate(line)
            end
            # puts "first_indices:"
            # puts first_indices.inspect
            # Compute the evolution of the indices for the first line.
            evolution = Numo::DFloat.new(@r_shape[1]).seq
            # Compute the evolution for the other lines.
            (@r_shape[0]-1).times do |i|
                line = Numo::DFloat.new(@r_shape[1]).seq+(i+1)*@m_shape[1]
                # puts "line="
                # puts line.inspect
                evolution = evolution.concatenate(line)
            end
            evolution.reshape!(@r_shape[0]*@r_shape[1],1)
            # puts "evolution:"
            # puts evolution.inspect
            # Compute the full indices.
            @indices = first_indices.tile(@r_shape[0]*@r_shape[1],1) +
                      evolution.tile(1,@f_shape[0]*@f_shape[1])
            # puts "@indices:"
            # puts @indices.inspect
            # puts "last indice: #{@indices[-1]}"
        end

        ## compute the convolution of two matrices.
        #  NOTE: the characteristics of the convolution (step, padding)
        #  are supposed to be done with the constructor.
        #  @param m the matrix to convolute
        #  @param f the filter
        #  @return the convolution result
        def call(m,f)
            # puts "m=#{m.inspect}"
            # puts "f=#{f.inspect}"
            return (m[@indices].dot(f.reshape(f.size,1))).reshape!(*@r_shape)
        end
    end


    # Class for a fast pooling.
    #
    # Also include a method for describing a pooling layer.
    #
    # @example assuming we want to make a max pooling of a matrix of
    #          shape (128,128) with pooler of shape (3x3) of matrix m:
    #          pool = Pool.new([128,128],[3,3],:max)
    #          pool.(m)
    # NOTE: the step and pad are not usable yet.
    class Pool

        ## Build a new pool algorithm.
        #  @param m_shape the shape of the matrices to covolute
        #  @param p_shape the shape of the pool
        #  @param func the function used for pooling
        def initialize(m_shape,p_shape,func = :max)
            @func = func
            # Get and check the shapes.
            @m_shape = m_shape.to_a[0..1]
            @p_shape = p_shape.to_a[0..1]
            # Compute the shape of the result.
            @r_shape = [@m_shape[0]/@p_shape[0],@m_shape[1]/@p_shape[1]]
            # Generate the matrix of indices for picking the elements for the
            # convolution.
            # Compute the indices for the first convolution product.
            first_indices = Numo::DFloat.new(@p_shape[1]).seq
            (@p_shape[0]-1).times do |i|
                line = Numo::DFloat.new(@p_shape[1]).seq+(i+1)*@m_shape[1]
                first_indices = first_indices.concatenate(line)
            end
            # puts "first_indices:"
            # puts first_indices.inspect
            # Compute the evolution of the indices for the first line.
            evolution = Numo::DFloat.new(@r_shape[1]).seq*@p_shape[1]
            # Compute the evolution for the other lines.
            (@r_shape[0]-1).times do |i|
                line = Numo::DFloat.new(@r_shape[1]).seq*@p_shape[1]+
                    (i+1)*@m_shape[1]*@p_shape[0]
                # puts "line="
                # puts line.inspect
                evolution = evolution.concatenate(line)
            end
            evolution.reshape!(@r_shape[0]*@r_shape[1],1)
            # puts "evolution:"
            # puts evolution.inspect
            # Compute the full indices.
            @indices = first_indices.tile(@r_shape[0]*@r_shape[1],1) +
                      evolution.tile(1,@p_shape[0]*@p_shape[1])
            # puts "@indices:"
            # puts @indices.inspect
            # puts "last indice: #{@indices[-1]}"

            # Compute the indices for applying the pooling function.
            @p_size = @p_shape[0] * @p_shape[1] 
            @r_size = @r_shape[0] * @r_shape[1] 
            # Generate the tile selection.
            sel_tile = (Numo::DFloat.new(@r_size).seq * @p_size)
            sel_tile.reshape!(@rsize,1)
            # puts "sel_tile: " ; puts sel_tile.inspect
            # Then the final selection indices.
            @sel_indices = []
            if @func != :mean then
                @p_size.times do |idx|
                    sel_line = Numo::DFloat.zeros(1,1)
                    sel_line[0] = idx
                    @sel_indices << (sel_line.tile(@r_size,1) + sel_tile)
                end
            end
            # The mean filter.
            @mean_filter = Numo::DFloat.ones(@p_size,1)
            @mean_filter = @mean_filter / @p_size
        end

        ## compute the pooling.
        #  @param m the matrix to convolute
        #  @return the convolution result
        def call(m)
            # Shape the m for fast pooling.
            m = m[@indices]
            if @func == :max || @func == :min then
                x = m[@sel_indices[0]]
                # puts "x=" ; puts x.inspect
                @sel_indices[1..-1].each do |indices|
                    y = m[indices]
                    # Max / min computation of x and y... without any comparison:
                    # max(x,y) = (x+y+abs(x-y))/2
                    # min(x,y) = (x+y-abs(x-y))/2
                    # This way can be applied at the matrix level for high
                    # performance.
                    x = (x + y + (@func ==:max ? (x-y).abs : -(x-y).abs))/2.0
                end
                return x.reshape!(*@r_shape)
            elsif @func == :mean then
                return (m.dot(@mean_filter)).reshape!(*@r_shape)
            else
                return Numo::DFloat.asarray(m.split(@r_shape[0]*@r_shape[1]).
                                            map(&@func)).reshape!(*@r_shape)
            end
        end

        ## Compute the reverse pooling.
        #  @param m    the matrix to reverse the pooling
        #  @param prev the former input matrix
        #  @return the result of the reverse pooling computation.
        def reverse(m,prev)
            # Expand m to the size of the input.
            m = m.repeat(@p_shape[0],axis:0).repeat(@p_shape[1],axis:1)
            m = m.insert(-1,0.0,axis:0) while m.shape[0] < prev.shape[0]
            m = m.insert(-1,0.0,axis:1) while m.shape[1] < prev.shape[1]
            # Process m.
            if @func == :max || @func == :min then
                # Create the resulting matrix: set to one the places where
                # values were selected and 0 otherwize.
                return m.eq(prev).cast_to(Numo::DFloat)
            elsif @func == :mean then
                # Divide by the sumber of elements used for the mean.
                return m / @p_size
            else
                raise "Pooling function: #{func} not supported yet."
            end
        end
    end


    
    

    # Classes describing NN layers.

    class Layer
        ## The feed forward computation.
        #  @param input the input matrix of the layer
        #  @return the forward computation result matrix
        def forward(input)
            raise "The feedforward method must be defined in: #{self.class}"
        end

        ## The back propagation.
        #  @param delta_in the input delta matrix of the layer
        #  @return the backward delta computation result matrix
        def backward(delta_in)
            raise "The backpropagate method must be defined in: #{self.class}"
        end

        ## Update the weights and biases.
        #  @param delta_in the input delta matrix of the layer
        def update(delta_in)
            raise "The update method must be defined in: #{self.class}"
        end

        ## Get the input geometry.
        def geo_in
            raise "The geo_in method must be defined in: #{self.class}"
        end

        ## Get the output geometry.
        def geo_out
            raise "The geo_out method must be defined in: #{self.class}"
        end

        ## Get the forward computation result.
        def output
            raise "The output method must be defined in: #{self.class}"
        end

        ## Get the backward computation result (delta).
        def delta
            raise "The delta method must be defined in: #{self.class}"
        end

        ## Returns the structure of the layer.
        def structure
            return { class: self.class,
                     input: self.geo_in, output: self.geo_out }
        end
    end

    
    ## Class for describing an activation layer.
    #
    #  Also includes a method for describing an activation layer.
    class Activation < Layer

        ## Build a new activation layer.
        #  @param geo_in the geometry of the inputs
        #  @param func the function to use as activation for each input.
        def initialize(geo_in, func)
            # Check set and the input geometry.
            @geo_in = geo_in.flatten.map(&:to_i)

            # Check and the function: it must be a pair function and derivate.
            @func = func.map {|e| e.to_proc }

            # The input geometry is also the output geometry.
            @geo_out = @geo_in

            # Initialize the inputs and output to nil for generating an error
            # in case of misuse.
            @input = nil
            @output = nil
        end

        ## The feed forward computation.
        #  @param input the input matrix of the layer
        #  @return the forward computation result matrix
        def forward(input)
            @input = input
            @output = @func[0].call(@input)
            return @output
        end

        ## The back propagation.
        #  @param delta_in the input delta matrix of the layer
        #  @return the backward delta computation result matrix
        def backward(delta_in)
            # puts "delta_in.shape=#{delta_in.shape}"
            # puts "@input.shape=#{@input.shape}"
            @delta = @func[1].call(@input,@output)*delta_in
        end

        ## Update the weights and biases.
        #  @param delta_in the input delta matrix of the layer
        def update(delta_in)
            # Nothing to do.
        end

        ## Get the input geometry.
        def geo_in
            return @geo_in.clone
        end

        ## Get the output geometry.
        def geo_out
            return @geo_out.clone
        end

        ## Get the forward computation result.
        def output
           return @output
        end

        ## Get the backward computation result (delta).
        def delta
           return @delta
        end


        ## Returns the structure of the layer.
        def structure
            return { class: self.class,
                     input: self.geo_in, output: self.geo_out,
                     func: @func,
                     parameters: 0 }
        end



        ## Describes a dense layer.
        #  @param func the activation function.
        def self.[](func)
            return [ Activation, func ]
        end

    end



    
    ## Class for describing a dense layer.
    #
    #  Also includes a method for describing a dense layer.
    class Dense < Layer

        ## Build a new dense layer.
        #  @param geo_in the geometry of the inputs
        #  @param size the number of neurons in the layer
        #  @param opt the options of the layer:
        #         :rand_weigth and :rand_bias for the randomization of the
        #         weights and biases, :rate for the learning rate.
        def initialize(geo_in, size, opt = {})
            # Check and set the input geometry.
            @geo_in = geo_in.flatten.map(&:to_i)
            @size_in = @geo_in.reduce(&:*)
            # Check and set the output geometry.
            @size_out = size.to_i
            @geo_out = [@size_out]

            # Compute the inner geometries.
            @geo_bias   = @geo_out
            @geo_weight = [@size_out, @size_in]
            @geo_delta  = [@size_in]

            # Compute the flag for reshaping.
            @reshape  = @geo_delta != @geo_in

            # Process the options.
            # Randomization.
            @rand_weight = opt[:rand_weight]
            @rand_weight = RAND_DEFAULT unless @rand_weight
            @rand_bias   = opt[:rand_bias]
            @rand_bias   = RAND_DEFAULT unless @rand_bias
            @rate        = opt[:rate]
            @rate        = @rate ? @rate.to_f : RATE_DEFAULT

            # Create the weights matrix.
            @weights = Numo::DFloat.zeros(@geo_weight)
            self.randomize_weights
            # Create the biases vector.
            @biases  = Numo::DFloat.zeros(@geo_bias)
            self.randomize_biases

            # Initialize the inputs and output to nil for generating an error
            # in case of misuse.
            @input = nil
            @output = nil
        end

        ## Randomize the weights.
        def randomize_weights
            @weights = @weights.map { |e| @rand_weight.call }
        end

        ## Randomize the biases.
        def randomize_biases
            @biases = @biases.map { |e| @rand_bias.call }
        end

        ## The feed forward computation.
        #  @param input the input matrix of the layer
        #  @return the forward computation result matrix
        def forward(input)
            # puts "input.shape=#{input.shape}, @geo_delta=#{@geo_delta}, @geo_in=#{@geo_in}"
            @input = @reshape ? input.reshape(*@geo_delta) : input
            @output = @weights.dot(@input)+@biases
            return @output
        end

        ## The back propagation.
        #  @param delta_in the input delta matrix of the layer
        #  @return the backward delta computation result matrix
        def backward(delta_in)
            # puts "delta_in=#{delta_in.inspect}"
            # @delta    = @weights.transpose.dot(delta_in)
            @delta = delta_in.dot(@weights)
            # puts "@delta=#{@delta.inspect}"
            return @reshape ? @delta.reshape(*@geo_in) : @delta
        end

        ## Update the weights and biases.
        #  @param delta_in the input delta matrix of the layer
        def update(delta_in)
            # puts "delta_in=#{delta_in.inspect}"
            # puts "@input=#{@input.inspect}"
            # puts "@biases=#{@biases.inspect}"
            # puts "delta_in.outer(@output)=#{delta_in.outer(@output).inspect}"
            # puts "@weights=#{@weights.inspect}"
            # @weights -= delta_in.dot(@output.transpose)*@rate
            @weights -= delta_in.outer(@input)*@rate
            @biases -= delta_in*@rate
        end

        ## Get the input geometry.
        def geo_in
            return @geo_in.clone
        end

        ## Get the output geometry.
        def geo_out
            return @geo_out.clone
        end

        ## Get the forward computation result.
        def output
           return @output
        end

        ## Get the backward computation result (delta).
        def delta
           return @delta
        end


        ## Returns the structure of the layer.
        def structure
            return { class: self.class, 
                     input: self.geo_in, output: self.geo_out,
                     weights: @weights.shape, biases: @biases.shape,
                     parameters: @weights.size*@biases.size }
        end



        ## Describes a dense layer.
        #  @param size the size of the layer (number of neurons).
        #  @param opt the options of the layer.
        def self.[](size, opt = {})
            return [ Dense, size.to_i, opt ]
        end

    end

    
    ## Class for describing a convolution layer.
    #
    #  Also includes a method for describing a convolution layer.
    class Convolution < Layer

        ## Build a new convolution layer.
        #  @param geo_in the geometry of the inputs
        #  @param geo_filter the geometry of the filter
        #  @param num the number of filters
        #  @param step the step size
        #  @param pad the pad size
        #  @param opt the options of the layer:
        #         :rand_weigth for the randomization of the weights and biases,
        #         :rate for the learning rate.
        def initialize(geo_in, geo_filter, num, step, pad, opt = {})
            # Check and set the input geometry.
            @geo_in = geo_in.flatten.map(&:to_i)
            @geo_in = [1,*@geo_in] if @geo_in.size < 3
            # Check and set the filter geometry.
            @geo_filter = geo_filter.flatten.map(&:to_i)
            
            # Check and set the number of filter, the step and the pad.
            @num = num.to_i
            @step = step.to_i
            @pad = pad.to_i

            # Get the number of channels
            @ch_in = @geo_in[0]
            @ch_out = @num*@ch_in

            # Compute the output geometry.
            @geo_out = [ @ch_out, 
                         (@geo_in[1]+(-(@geo_filter[0]/2)+@pad)*2)/@step,
                         (@geo_in[2]+(-(@geo_filter[1]/2)+@pad)*2)/@step ]
            @geo_out[1] += 1 if @geo_filter[0].even?
            @geo_out[1] += 1 if @geo_filter[1].even?

            # Compute the reverse filters geometry
            @geo_reverse = [ @geo_filter[0]+(@geo_out[1]-1)*2,
                             @geo_filter[1]+(@geo_out[2]-1)*2 ]

            # Process the options.
            # Randomization.
            @rand_weight = opt[:rand_weight]
            @rand_weight = RAND_DEFAULT unless @rand_weights
            @rate        = opt[:rate]
            @rate        = @rate ? @rate.to_f : RATE_DEFAULT

            # Create the filters' matrices.
            @filters = @ch_out.times.map { Numo::DFloat.zeros(@geo_filter) }
            self.randomize_filters

            # Create the reverse filters' matrices.
            @r_filters = @ch_out.times.map do
                Numo::DFloat.zeros(@geo_reverse)
            end

            # Create the convolution algorithm for the forward pass.
            @conv_forward = Convolve.new(@geo_in[1..2],@geo_filter,step,pad)
            # Create the convolution algorithm for the backward pass.
            @conv_backward =Convolve.new(@geo_reverse,@geo_out[1..2],step,pad)
            # Create the convolution algorithm for updating the filter.
            @conv_update =  Convolve.new(@geo_in[1..2],@geo_out[1..2],step,pad)
        end

        ## Randomize the filters.
        def randomize_filters
            @filters = @filters.map do |filter|
                filter.map { @rand_weight.call }
            end
        end

        ## The feed forward computation.
        #  @param input the input matrix of the layer
        #  @return the forward computation result matrix
        def forward(input)
            @input_array = input.ndim == 3 ? input.each_over_axis.to_a : [input]
            @output_array = []
            pos = 0
            @ch_in.times do |i|
                @num.times do
                    @output_array << @conv_forward.(@input_array[i],@filters[pos])
                    pos += 1
                end
            end
            @output = Numo::DFloat.asarray(@output_array)
        end


        # ## Convert a filter for full convolution in reverse.
        # #  @param f the filter to convert.
        # #  @param d the dimension of the block that will convolve the filter.
        # def reverse_filter(f,d)
        #     # First rotate the filter.
        #     f = f.fliplr.flipud
        #     # The expand its size for full convolution.
        #     (d[0]-1).times do
        #         f = f.insert(0, 0.0, axis: 0)
        #         f = f.insert(-1,0.0, axis: 0)
        #     end
        #     (d[1]-1).times do
        #         f = f.insert(0, 0.0, axis: 1)
        #         f = f.insert(-1,0.0, axis: 1)
        #     end
        #     return f
        # end

        ## Compute the reverse filters for full convolution of the delta.
        def reverse_filters
            # puts "@r_filters[0].shape=#{@r_filters[0].shape}"
            # puts "@geo_out[1]-1=#{@geo_out[1]-1}"
            # puts "@geo_out[1]-1+@geo_filter[0]=#{@geo_out[1]-1+@geo_filter[0]}"
            @filters.each_with_index do |f,i|
                @r_filters[i][(@geo_out[1]-1)..(@geo_out[1]-2+@geo_filter[0]),
                          (@geo_out[2]-1)..(@geo_out[2]-2+@geo_filter[1])] =
                f.fliplr.flipud
                # f
            end
        end


        ## The back propagation.
        #  @param delta_in the input delta matrix of the layer
        #  @return the backward delta computation result matrix
        def backward(delta_in)
            @delta = []
            # Compute the reverse filters.
            self.reverse_filters
            # @r_filters = @filters.map do |f|
            #     # self.reverse_filter(f,@geo_out[1..2])
            #     Numo::DFloat.zeros(@geo_filter[0]+(@geo_out[1]-1)*2,@geo_filter[1]+(@geo_out[2]-1)*2)
            # end
            # puts "Reverse filter shape: #{@r_filters[0].shape}"
            # puts "geo_in=#{@geo_in} geo_out=#{@geo_out}"
            # Apply them
            sdelta = []
            pos = 0
            delta_in.each_over_axis do |d|
                sdelta << @conv_backward.(@r_filters[pos],d)
                # puts "d.shape=#{d.shape} sdelta[-1].shape=#{sdelta[-1].shape}"
                pos = pos+1
                if pos == @num then
                    @delta = Numo::DFloat.asarray(sdelta).mean
                    sdelta = []
                    pos = 0
                end
            end
            @delta = Numo::DFloat.asarray(@delta)
            # puts "@delta.shape=#{@delta.shape} @geo_in=#{@geo_in}"
        end

        ## Update the weights.
        #  @param delta_in the input delta matrix of the layer
        def update(delta_in)
            # puts "delta_in.shape=#{delta_in.shape}"
            # puts "input.shape=#{@input_array[0].shape}"
            delta_in = delta_in.each_over_axis.to_a
            @filters = @filters.map.with_index do |filter,idx|
                # puts "delta_in[idx]=#{delta_in[idx].inspect}"
                # puts "filter=#{filter.inspect}"
                # puts "update=#{@conv_update.(@input_array[idx/@num],delta_in[idx]).inspect}"
                filter - 
                    @conv_update.(@input_array[idx/@num],delta_in[idx])*@rate
            end
        end

        ## Get the input geometry.
        def geo_in
            return @geo_in.clone
        end

        ## Get the output geometry.
        def geo_out
            return @geo_out.clone
        end


        ## Get the forward computation result.
        def output
           return @output
        end

        ## Get the backward computation result (delta).
        def delta
           return @delta
        end


        ## Returns the structure of the layer.
        def structure
            return { class: self.class,
                     input: self.geo_in, output: self.geo_out,
                     filter_size: @geo_filter, filter_num: @num,
                     step: @step, pad: @pad }
        end


        ## Describes a convolution layer.
        #  @params args list of arguments as follows:
        #  Convolution[filter_width, filter_height, filter_num, pad_size,
        #     step_size, opt] or
        #    Convolution[size: [filter_width,filter_height], num: filter_num,
        #     step: step_size, pad: pad_size, opt: opt]
        #    where pad and step have default value to 0 and 1, respectively,
        #    and opt is by default to {}.
        #    When the filter is square [filter_width,filter_height] can be
        #    abbreviated to filter_size.
        def self.[](*args)
            res = [Convolution]
            if args[0].is_a?(Hash) && args.size == 1 then
                # Hash arguments.
                args = args[0]
                # Check and set each parameter of the convolution layer.
                # Size of the filters.
                size = args[:size]
                raise "Convolution layer without filter size." unless size
                if size.is_a?(Array) then
                    raise "Invalid convolution layer size: #{size}" unless size.size == 2
                    res[1] = size.map {|i| i.to_s }
                else
                    res[1] = [size.to_i, size.to_i]
                end
                # Number of filters
                num = args[:num]
                raise "Convolution layer without number of filters." unless num
                res[2] = num.to_i
                # Step
                step = args[:step]
                step = step ? step.to_i : 1
                res[3] = step
                # Pad
                pad = args[:pad]
                pad = pad ? pad.to_i : 0
                res[4] = pad
                # Options
                opt = args[:opt]
                res[5] = opt.to_h if opt
            else
                # Size of the filters.
                raise "Convolution layer without filter size." unless args.size >= 2 
                res[1] = [ args[0].to_i, args[1].to_i ]
                # Number of filters.
                raise "Convolution layer without number of filters." unless args.size >= 3 
                res[2]  = args[2].to_i
                # Step
                step = args[3]
                step = step ? step.to_i : 1
                res[3] = step
                # Pad
                pad = args[4]
                pad = pad ? pad.to_i : 0
                res[4] = pad
                # Options
                opt = args[5]
                res[5] = opt.to_h if opt
            end
            return res
        end
    end


    ## Class for describing a convolution layer.
    #
    #  Also includes a method for describing a convolution layer.
    class Pooling < Layer

        ## Build a new pooling layer.
        #  @param geo_in the geometry of the inputs
        #  @param geo_pool the geometry of the pooler
        #  @param func the type of pooling: :max, :min or :mean
        #  @param opt the options of the layer: for now none.
        def initialize(geo_in, geo_pool, func, opt = {})
            # Check and set the input geometry.
            @geo_in = geo_in.flatten.map(&:to_i)
            @geo_in = [1,*@geo_in] if @geo_in.size < 3
            # Check and set the pooler geometry.
            @geo_pool = geo_pool.flatten.map(&:to_i)

            # Check and set the type of pooling.
            @func = func.to_sym
            
            # Get the number of channels
            @ch_in = @geo_in[0]
            @ch_out = @ch_in

            # Compute the output geometry.
            @geo_out = [ @ch_out, 
                         @geo_in[1]/@geo_pool[0],
                         @geo_in[2]/@geo_pool[1] ]

            # Create the pooling algorithm.
            @pooler = Pool.new(@geo_in[1..2],@geo_pool,@func)
        end

        ## The feed forward computation.
        #  @param input the input matrix of the layer
        #  @return the forward computation result matrix
        def forward(input)
            @input_array = input.ndim == 3 ? input.each_over_axis.to_a : [input]
            @output_array = @input_array.map { |ch| @pooler.(ch) }
            @output = Numo::DFloat.asarray(@output_array)
        end

        ## The back propagation.
        #  @param delta_in the input delta matrix of the layer
        #  @return the backward delta computation result matrix
        def backward(delta_in)
            @delta = []
            delta_in.each_over_axis.with_index do |ch,idx|
                @delta << @pooler.reverse(ch,@input_array[idx])
            end
            @delta = Numo::DFloat.asarray(@delta)
        end

        ## Update the weights.
        #  @param delta_in the input delta matrix of the layer
        def update(delta_in)
            # Nothing to do.
        end

        ## Get the input geometry.
        def geo_in
            return @geo_in.clone
        end

        ## Get the output geometry.
        def geo_out
            return @geo_out.clone
        end


        ## Get the forward computation result.
        def output
           return @output
        end

        ## Get the backward computation result (delta).
        def delta
           return @delta
        end


        ## Returns the structure of the layer.
        def structure
            return { class: self.class,
                     input: self.geo_in, output: self.geo_out,
                     pool_size: @geo_pool, func: @func }
        end


        ## Describes a pooling layer.
        #  @params args list of arguments as follows:
        #  Pooling[pool_width, pool_height, func = :max, opt = {}] or
        #  Pooling[size: [pool_width,pool_height], func: func, opt: opt]
        #  where pad and step have default value to 0 and 1, respectively,
        #  and opt is by default to {}.
        #  When the pooler is square [pool_width,pool_height] can be
        #  abbreviated to pool_size.
        def self.[](*args)
            res = [Pooling]
            if args[0].is_a?(Hash) && args.size == 1 then
                # Hash arguments.
                args = args[0]
                # Check and set each parameter of the pooling layer.
                # Size of the filters.
                size = args[:size]
                raise "Pooling layer without filter size." unless size
                if size.is_a?(Array) then
                    raise "Invalid pooling layer size: #{size}" unless size.size == 2
                    res[1] = size.map {|i| i.to_s }
                else
                    res[1] = [size.to_i, size.to_i]
                end
                # Function
                func = args[:func]
                func = func ? func.to_sym : :max
                res[2] = func
                # Options
                opt = args[:opt]
                res[3] = opt.to_h if opt
            else
                # Size of the Pooler.
                raise "Pooling layer without pooler size." unless args.size >= 2 
                res[1] = [ args[0].to_i, args[1].to_i ]
                # Type of pooler.
                func = args[2]
                func = func ? func.to_sym : :max
                res[2] = func
                # Options
                opt = args[3]
                res[3] = opt.to_h if opt
            end
            return res
        end

    end





    ## Describes a convulotion neural network based on backpropagation.
    #  @since 1.0.0
    class CNN
        ## constructor <br>
        #  Creates a CNN from the description given as argument. <br>
        #  @param geo_in the input geometry.
        #  @param args description of each layer of the NN.
        #  Each argument element can be as followings:
        #  * Dense[size] for a dense layer with size neurons.
        #  * Convolution[filter_width, filter_height, filter_num, pad_size,
        #     step_size] or
        #    Convolution[size: [filter_width,filter_height, num: filter_num,
        #     pad: pad_size, step: step_size] for a convolution layer,
        #    where pad and step have default values to 0 and 1.
        #    When the filter is square [filter_width,filter_height] can be
        #    abbreviated to filter_size.
        #  * Pooling[pool_width, pool_height] for a pooling layer.
        #    When the pooling matrix is square [pool_width,pool_height] can be
        #    abbreviated to pool_size only.
        #  * A activation function layer among the followings:
        #    Linear, Sigmoid, Tanh, ReLU, LeakyReLU, ELU, SELU, Softplus, Swish,
        #    Mish, Softmax.
        #  * The following options:
        #    loss: the loss function: MeanSquare or CrossEnthropy
        #    rate: the learing rate (float)
        #
        #  NOTE: each layer can have some options amongs the following:
        #  * Dense: :rand_weight and :rand_bias for the randomization of the
        #           weights and biases
        #  * Convolution: :rand_weight for the randomization of the weights
        #
        #  @example initialization of the neural network
        #   nn = FastNeurons::CNN.new([28,28],[Convolution[3,3,32],ReLU,Pool[2,2],Dense[10],Sigmoid], rate: 0.001)
        #   nn = FastNeurons::CNN.new([28,28],[Convolution[size: [3,3], num: 32, pad: 1], ReLU, Pool[2,2], Dense[10], Sigmoid], loss: CrossEntropy)
        #  @since 1.0.0
        def initialize(geo_in, layers, opt = {})
            # Check and set the input geometry.
            @geo_in = geo_in.map {|d| d.to_i }

            # Process the options.
            @rate = opt[:rate]
            @rate = @rate ? @rate.to_f : RATE_DEFAULT
            @loss_func = opt[:loss]
            @loss_func = MeanSquare unless @loss_func

            # Build the NN structure.
            geo = @geo_in # The current geometry
            @layers = layers.map do |layer|
                klass = layer.shift
                layer = klass.new(geo,*layer)
                geo = layer.geo_out
                layer
            end

            # Set the output geometry.
            @geo_out = @layers[-1].geo_out

            # The current input, output, expected output and final delta.
            @input  = Numo::DFloat.zeros(@geo_in)
            @output = Numo::DFloat.zeros(@geo_out)
            @expect = Numo::DFloat.zeros(@geo_out)
            @delta  = Numo::DFloat.zeros(@geo_in)
            # Initialize the loss and loss gradian computations.
            self.reset_loss
            self.reset_grad
        end


        ## Reset the loss computation.
        def reset_loss
            @loss = 0.0
            @loss_num = 0
        end

        ## Reset the gradian computation.
        def reset_grad
            @grad = 0.0
            @grad_num = 0.0
        end

        ## Update the loss.
        #  @return the current loss value.
        def update_loss
            @loss += @loss_func[0].call(@output,@expect)
            @loss_num += 1
            return @loss
        end

        ## Update the loss gradian.
        #  @return the current loss gradian value.
        def update_grad
            @grad += @loss_func[1].call(@output,@expect)
            @grad_num += 1
            return @grad
        end

        ## Update the input.
        #  @param input the input to set.
        def update_input(input)
            # Check and set the input.
            @input = input.shape==@geo_in ? input : input.reshape(*@geo_in)
        end

        ## Update the expected output.
        #  @param expect the expected output to set.
        def update_expect(expect)
            # Check and set the expected output.
            @expect = expect.shape==@geo_out ? expect : expect.reshape(*@geo_out)
        end


        ## The feed forward computation.
        #  @return the forward computation result matrix
        def forward
            vector = @input
            @layers.each { |layer| vector = layer.forward(vector) }
            @output = vector
        end

        ## The back propagation.
        #  @return the backward delta computation result matrix
        def backward
            # Backpropagate.
            vector = @grad / @grad_num
            @layers[1..-1].reverse_each do |layer| 
                layer.update(vector)
                vector = layer.backward(vector)
            end
            @layers[0].update(vector)
            # # Set the final delta.
            # @delta = vector
        end

        ## Get the current output.
        def output
            return @output
        end

        ## Get the current loss.
        def loss
            return @loss
        end

        ## Get the current loss gradian.
        def grad
            return @grad
        end

        ## Get the current delta.
        def delta
            return @delta
        end

        ## Get the structure of the neural network.
        def structure
            return @layers.map {|layer| layer.structure }
        end

    end



end
