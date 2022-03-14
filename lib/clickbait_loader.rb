require 'zlib'

# Loader of MNIST.
# @version 1.1.0
# @since 1.0.0
class ClickBaitLoader
    # constructor
    # @param [String] clickbait_path path of the clickbait texts.
    # @param [String] non_clickbait_path path of the non-clickbait texts.
    # @since 1.0.0
    def initialize(clickbait_path, non_clickbait_path)
        @cb_path  = clickbait_path
        @ncb_path = non_clickbait_path
    end

    # Get the clickbait strings.
    def get_clickbait_strings
        return @cb_strs
    end

    # Get the non-clickbait strings.
    def get_non_clickbait_strings
        return @ncb_strs
    end

    # Get the clickbait vectors.
    def get_clickbait_vectors
        return @cb_vecs
    end

    # Get the non-clickbait vectors.
    def get_non_clickbait_vectors
        return @ncb_vecs
    end

    # Load clickbait texts.
    # @since 1.0.0
    def load_clickbaits
        @cb_strs, @cb_vecs = self.load_statements(@cb_path)
    end

    # Load non-clickbait texts
    # @since 1.0.0
    def load_non_clickbaits
        @ncb_strs, @ncb_vecs = self.load_statements(@ncb_path)
    end

    # Load a set of statements in a gz file.
    # @param [String] path file path.
    def load_statements(path)
        # Load the full text
        txt = self.load_gzipped_idx_file(path)
        # Get the statements from the text.
        stm_strs = self.get_statements(txt)
        # Vectorize the characters.
        stm_vecs = stm_strs.map { |stm| self.vectorize(stm) }
        # Return the result.
        return [ stm_strs, stm_vecs ]
    end

    # Extract the statements from a text.
    # @param [String] txt the text to process.
    def get_statements(txt)
        # Split with new line.
        strs = txt.split("\n")
        # Remove the empty statements.
        strs.delete("")
        # Return the result.
        return strs
    end

    # The vectorization table.
    CHARS = [" ", "!", '"', "'", ",", "."] + ("0".."9").to_a + [":", ";"] +
            ("A".."Z").to_a + ("a".."z").to_a
    CHAR2VECTOR = Hash.new([0.0] * CHARS.size)
    CHARS.each_with_index do |c,i|
        CHAR2VECTOR[c] = [0.0] * CHARS.size
        CHAR2VECTOR[c][i] = 1.0
    end

    # Vectorize a string statement.
    # @param [String] str the string statement to convert to a list of vectors.
    def vectorize(str)
        return str.each_char.map { |c| CHAR2VECTOR[c] }
    end

    # Load gzipped file.
    # @param [String] path file path
    # @since 1.0.0
    def load_gzipped_idx_file(path)
        file = File.open(path,"rb")
        stream = Zlib::GzipReader.new(file)
        return stream.read
    end

    # Normalize pixel values to a continuous values from 0 ~ 255 to 0 ~ 1.
    # @param [Array] inputs array of pixel values.
    # @param [Range] scaling range for the result.
    # @return [Array] array of normalized pixel values
    # @since 1.0.0
    def normalize(inputs, range = 0.0..1.0)
        return inputs.map{|pixel|
            (pixel/256.0) * (range.last-range.first) + range.first
        }
    end

    # Binarize pixel values to a continuous values from 0 ~ 255 to 0,1.
    # @param [Array] inputs array of pixel values.
    # @return [Array] array of binarized pixel values
    # @since 1.1.0
    def binarize(inputs, min = 0.0, max = 1.0)
        return inputs.map{|pixel|
            # pixel > 0.5 ? 1.0 : 0.0
            pixel > ((max-min)/2.0+min) ? max : min
        }
    end

    # Print ascii of MNIST.
    # @param [Array] inputs array of pixel values
    # @param [Range] the scaling range on the inputs.
    # @since 1.0.0
    def print_ascii(inputs, range = 0.0..1.0)
        # inputs = inputs.to_a.flatten.map {|pixel| pixel*255}
        inputs = inputs.to_a.flatten.map {|pixel|
            ((pixel-range.first)/(range.last-range.first))*255
        }
        outputs = inputs.each_slice(28).map do |row|
            row.map do |darkness|
                darkness < 64 ?  " " : ( darkness < 128 ? "." : "X" )
            end.join
        end.join("\n")
        puts outputs
    end
end
