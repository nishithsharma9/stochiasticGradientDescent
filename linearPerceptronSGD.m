function [optimumTheta] = linearPerceptronSGD(data,tolerance)

% Split the data into x feature and y output class
x = data(:,1:2);
y = data(:,3);

% Add the column of ones corresponding to theta0
x = [(ones(length(x),1)),x];

% t will store the total number of updates we do on theta
t = 0; 

% Initialize theta_old and theta_new
theta_old = zeros(1,width(x));
theta_new = ones(1,width(x));

% For the initial theta_new assumed, calculate the misclassfication
% percentage.
misinter = y.*x*theta_new';
misinter(misinter>=0) = 0;
misinter(misinter<0) = 1;
%Calculate misclassification percentage for theta_new over whole dataset
percentageMisclassification = sum(misinter)*100/length(x);

% i is used to rotate between {1,2....N}
i= 1;

% This just helps to update i using iter%N
iter=0;

% This vector we will  store the misclassification percentage at each update
% of theta.
misclassification = [];

% Continue to loop through till either of the condition fails:
%1. norm becomes lesser than tolerance
%2. misclassification percentage becomes 0, All are classified correctly
while (norm(theta_new-theta_old)>tolerance) & (percentageMisclassification>0)
    % Condition to check if misclassified then enter the 
    % block to update theta
    if (y(i)'.*x(i,:)*theta_new' < 0)
        % Theta update
        theta_old = theta_new;
        theta_new = theta_new + 0.1*y(i)*x(i,:);
        % Update t for each theta update
        t=t+1;

        % For the updated theta, we need to check how many misclassification
        % we have done
        misinter = y.*x*theta_new';
        misinter(misinter>=0) = 0;
        misinter(misinter<0) = 1;
        
        %Calculate misclassification percentage for theta over whole dataset
        percentageMisclassification = sum(misinter)*100/length(x);
        
        % Store all the misclassification percentage in a vector for each
        % theta update
        misclassification = [misclassification; percentageMisclassification];
    end

    %update iter to keep updating i to iterate over the dataset
    iter = iter + 1;
    i = mod(iter,length(x))+1;
end

%Display the results
theta_new
diffNorm = norm(theta_new-theta_old)
t
percentageMisclassification

% Plot Graph

% Plot of the change in misclassification over the t iterations,
% for each update we calculate and store the misclassification percentage.
figure('Name','Misclassidication percentage over number of iterations t')
plot(misclassification)
xlabel('Iterations')
ylabel('Percentage misinterpreted data points')

%Plot the data along with the model
figure('Name','Data Plot & Classifier')
xlabel('Iterations')
ylabel('Percentage misinterpreted data points')
for i = 1:length(x)
    if (y(i)==1)
        plot(x(i,2),x(i,3),'o','MarkerFaceColor','red')
        hold on
    else
        plot(x(i,2),x(i,3),'o','MarkerFaceColor','green')
        hold on
    end
end
hold on

% To Plot the classfication model
x1 = 0 :0.1: 1;
x2 = (-theta_new(1) - x1*theta_new(2))/theta_new(3);
plot(x1,x2,'color','blue')
xlabel('X1 Feature')
ylabel('X2 Feature')
legend('y = 1','y = -1')