class OdeSystem:
    def __init__(self, equation):
        # Init some variables
        self.order = 0
        self.len_batch = 0
        self.input_data = None
        self.curr_model = None

        self.equation = equation
        self.process_equation()

    def process_equation(self):
        '''Function determines if this is a first or second order ODE
        '''
        split_eq = self.equation.replace(" ", "").split("=")
        self.eq = split_eq[1]

        order = 0
        for c in list(split_eq[0]):
            if c in ["x", "'"]:
                if c == "'":
                    order += 1
            else:
                tb = sys.exc_info()[2]
                raise SyntaxError('Expected x or \' , got "{}" instead.'.format(c)).with_traceback(tb)
        
        if order not in (1,2):
            tb = sys.exc_info()[2]
            raise ValueError('Expected order 1 or 2 equation, got order {} instead.'.format(order)).with_traceback(tb)

        self.order = order

    def create_input_data(self, num_batch = 15, len_batch = 1):
        '''Creates data that will be the input to the neural net. This is the x-data.
        Data will be shaped to (num_batch,len_batch). More elements means
        longer training time, but a more continuous line when fit.\nData is created
        between 0 and 1 to prevent scaling issues with the neural network.
        Args:
            num_batch: The number of batches. Default: 15
            len_batch: Number of values in each batch. Recommended to keep this
                       set to default for best results. Default: 1
        '''
        self.len_batch = len_batch
        n = num_batch * len_batch       # Number of total points
        input_data = np.linspace(0,1,n)
        input_data = input_data.reshape(num_batch,len_batch)

        self.input_data = input_data

    def solve(self, ic = 0, ic2 = None, epochs = 500, learning_rate = .01, nodes = 50):
        '''Temp
        '''
        self.ic = ic
        self.ic2 = ic2

        if (self.order == 1):
            fo_class = FirstOrder()
            self.curr_model = fo_class.fo(ic, epochs, learning_rate, nodes, 
                        self.input_data, self.len_batch, eq = self.eq)
        elif (self.order == 2):
            so_class = SecondOrder()
            self.curr_model = so_class.so(ic, ic2, epochs, learning_rate, nodes,
                        self.input_data, self.len_batch, eq = self.eq)
        else:
            raise ValueError("Something went wrong.")
            
            #print('Warning: order of ODE = {}, but a second initial condition was given. \
            #      The second initial condition will be ignored.'.format(ic2))

#        if (self.order == 2) & (ic2 != None):
#            pass
#        else:
#            raise TypeError('Expected an integer second initial condition; {} given instead'.\
#                            format(ic2))

    def predict(self, ax = None):
        '''Takes a trained model and plots the results with the input data.
        Args:
            ax: The subplot axis to plot the data onto. If None, it creates
                its own ax to plot to. Default: None
        Returns: 
        '''
        x_test = self.input_data
        prediction = self.curr_model.predict(x_test)
        prediction = prediction.reshape(1,-1)[0]
        x_test = x_test.reshape(1,-1)[0]

        results = TrialFunctions.first_order(self.ic, x_test, prediction)

        if ax == None:
            fig, ax = plt.subplots()

        ax.plot(x_test, results, label = 'Neural ODE', color = 'tomato')
        ax.scatter(x_test, results, color = 'tomato', s = 7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title("${}$".format(self.equation))
        ax.legend()
        
class FirstOrder:
    '''Solves a first order ODE; a different class
    is constructed for solving second order ODEs.
    '''
    def fo(self, ic, epochs, lr, nodes, input_data, len_batch, eq):
        self.bc = ic
        self.len_batch = len_batch
        self.input_data = input_data
        self.eq = eq

        # Calls the solve method with actually sovles
        # Uses much of the other functions here. 
        return self.solve(epochs, lr, nodes) # Returns a keras model

    def trial_func(self, t, x_r):
        '''This constructs a trial function for the ODE that 
        has the correct initial conditions. t is the time input
        series data; x_r is the neural net output. 
        '''
        func = self.bc +t*x_r
        return func
    
    def right_side(self, trial):
        '''Calculates the right side of the differential equation.
        Puts the output of the trial function (the prediction) into the 
        user inputted equation. The static method Parser.to_expression changes
        the user inputted string into a executable form in a safe manner.
        '''
        print("self.eq: {}".format(self.eq))
        right_side = Parser.to_expression(self.eq, trial)
        return right_side

    def loss_wrapper_fo(self, input_tensor):
        def loss_function_fo(y, y_pred):
            ## Find the trial solution and right-side
            trial = self.trial_func(input_tensor, y_pred)
            right = self.right_side(trial)

            ## Find left side by computing second derivitive of trial
            left = tf.gradients(trial, input_tensor)[0]

            ## Loss is the mean squared difference of left and right
            loss = tf.reduce_mean(tf.math.squared_difference(left, right))
            return loss
        return loss_function_fo

    def solve(self, epochs = 500, lr = .01, n_hidden_layer = 50):
        '''
        Given data reshaped to (n_batches, len_batch)
        Returns model after training epochs
        '''
        input_tensor = Input(shape=(self.len_batch,))
        hidden = Dense(n_hidden_layer, activation='tanh',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_tensor)
        hidden2 = Dense(n_hidden_layer, activation='tanh',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden)
        hidden3 = Dense(n_hidden_layer, activation='tanh',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden2)
        out = Dense(self.len_batch)(hidden3)

        model = Model(input_tensor, out)

        gradient_descent = optimizers.adam(learning_rate=lr)
        model.compile(loss = self.loss_wrapper_fo(input_tensor), optimizer = gradient_descent)
        model.fit(self.input_data, np.zeros((self.input_data.shape[0])), epochs = epochs)

        return model
        
        
class SecondOrder:
    '''Solves a first order ODE; a different class
    is constructed for solving second order ODEs.
    '''
    def so(self, ic, ic2, epochs, lr, nodes, input_data, len_batch, eq):
        self.bc = ic
        self.bc2 = ic2
        self.len_batch = len_batch
        self.input_data = input_data
        self.eq = eq

        # Calls the solve method with actually sovles
        # Uses much of the other functions here. 
        return self.solve(epochs, lr, nodes) # Returns a keras model

    def trial_func_x(self, t, x_r):
        '''This constructs a trial function for the ODE that 
        has the correct initial conditions. t is the time input
        series data; x_r is the neural net output. 
        '''
        func = self.bc + t*x_r
        return func

    def trial_func_v(self, t, x_r):
        func = self.bc2 + t*x_r
        return func
    
    def right_side(self, trial):
        '''Calculates the right side of the differential equation.
        Puts the output of the trial function (the prediction) into the 
        user inputted equation. The static method Parser.to_expression changes
        the user inputted string into a executable form in a safe manner.
        '''
        right_side = Parser.to_expression(self.eq, trial)
        return right_side

    def loss_wrapper_so(self, input_tensor):
        def loss_function_so(y, y_pred):
            '''
            '''
            # And now everything for x:
            trial = self.trial_func_x(input_tensor, y_pred)
            right = self.right_side(trial)
            left_x = tf.gradients(tf.gradients(trial, input_tensor)[0], input_tensor)[0]
            loss_x = tf.reduce_mean(tf.math.squared_difference(left_x, right))

            # Everything for v
            # In the case of, v = x' = trial'
            v0 = tf.gradients(trial, input_tensor)[0][0]
            print(v0)
            return loss_x + tf.math.squared_difference(v0, self.bc2)
        return loss_function_so

    def solve(self, epochs = 500, lr = .01, n_hidden_layer = 50):
        '''
        Given data reshaped to (n_batches, len_batch)
        Returns model after training epochs
        '''
        input_tensor = Input(shape=(self.len_batch,))
        hidden = Dense(n_hidden_layer, activation='tanh',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_tensor)
        hidden2 = Dense(n_hidden_layer, activation='tanh',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden)
        hidden3 = Dense(n_hidden_layer, activation='tanh',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(hidden2)
        out = Dense(self.len_batch)(hidden3)

        model = Model(input_tensor, out)

        gradient_descent = optimizers.adam(learning_rate=lr)
        model.compile(loss = self.loss_wrapper_so(input_tensor), optimizer = gradient_descent)
        model.fit(self.input_data, np.zeros((self.input_data.shape[0])), epochs = epochs)

        return model
   
   
class TrialFunctions(object):
    @staticmethod
    def first_order(ic, t, prediction):
        func = ic + t*prediction
        return func