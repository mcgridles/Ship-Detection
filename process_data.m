% TODO
%
% 1. Examine data
% 2. Perform PCA analysis
% 3. Create regression model
% 4. Create classification model?
%

start_date = datetime('01/01/2001');
end_date = datetime('12/31/2017');
num_days = days(end_date-start_date);

cfilename = 'Chicago-Crime-Analysis/crimes-in-chicago/cleaned_data.csv';
wfilename = 'Chicago-Crime-Analysis/chicago_weather.csv';

crime = readtable(cfilename,'DatetimeType','text');

crime.Properties.VariableNames(1) = {'Arrest'};
crime.Date = convertDates(crime.Date);
crime_cleaned = getCrimesPerDay(crime, start_date, num_days);

weather = readtable(wfilename,'DatetimeType','text');

weather.Properties.VariableNames(1) = {'STATION'};
weather.DATE = convertDates(weather.DATE);
weather.TMAX = str2double(weather.TMAX);
weather.TMIN = str2double(weather.TMIN);
weather.TOBS = str2double(weather.TOBS);
weather.WT01 = str2double(weather.WT01);
weather.WT03 = str2double(weather.WT03);
weather.WT04 = str2double(weather.WT04);
weather.WT05 = str2double(weather.WT05);
weather.WT06 = str2double(weather.WT06);
weather.WT11 = str2double(weather.WT11);

weather_cleaned = getWeatherPerDay(weather, start_date, num_days);

data = join(crime_cleaned, weather_cleaned);
writetable(data, 'Chicago-Crime-Analysis/data.csv');

% Dates are incorrectly converted to datetime arrays when loaded because
% they are in the form mm/dd/yy instead of mm/dd/yyyy
function [converted] = convertDates(dates)
    warning('off', 'MATLAB:datetime:AmbiguousDateString');
    
    % Dates are converted to:
    % 1. A serial number,
    % 2. A date string to remove the time component,
    % 3. A datetime array
    converted = datetime(datestr(datenum(dates),'mm/dd/yyyy'));
end

function [crime_out] = getCrimesPerDay(crime_in, start_date, num_days)
    crime_out = array2table(zeros(0,2),'VariableNames',{'Date', 'Crimes'});
    current_date = start_date;
    for i=1:num_days
        date_mask = (crime_in.Date == current_date);
        crimes_on_date = crime_in(date_mask,:);
        
        crime_out.Date(i) = datenum(current_date);
        crime_out.Crimes(i) = height(crimes_on_date);
        
        current_date = current_date + 1;
    end
    
    crime_out.Date = datetime(crime_out.Date, 'ConvertFrom', 'datenum');
end

function [weather_out] = getWeatherPerDay(weather_in, start_date, num_days)
    weather_out = array2table(zeros(0,11), 'VariableNames', ...
        {'Date', 'MaxTemp', 'MinTemp', 'AvgTemp', 'WT00', 'WT01',...
        'WT03', 'WT04', 'WT05', 'WT06', 'WT11'});
    
    current_date = start_date;
    for i=1:num_days
        date_mask = (weather_in.DATE == current_date);
        weather_on_date = weather_in(date_mask,:);
        
        if (~isempty(weather_on_date))
            weather_out.Date(i) = datenum(current_date);
            weather_out.MaxTemp(i) = weather_on_date.TMAX;
            weather_out.MinTemp(i) = weather_on_date.TMIN;
            weather_out.AvgTemp(i) = round((weather_on_date.TMAX...
                + weather_on_date.TMIN) / 2);
            
            weather_types = weather_on_date{1,7:12};
            weather_types(isnan(weather_types)) = 0;
            
            weather_out.WT00(i) = ~sum(weather_types);
            weather_out{i,6:end} = weather_types;
        end
        
        current_date = current_date + 1;
    end

    weather_out.Date = datetime(weather_out.Date, 'ConvertFrom', 'datenum');
end