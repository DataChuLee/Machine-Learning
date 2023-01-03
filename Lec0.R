install.packages("tidymodels")
library(tidymodels)

# Helper packages
install.packages('nycflights13')
install.packages("skimr")
library(nycflights13)    # for flight data
library(skimr)           # for variable summaries


flights %>% glimpse() 
flight_data %>% glimpse()
?flights

weather %>% glimpse()

set.seed(123)

flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = lubridate::as_date(time_hour)
  ) %>% 
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)


flight_data %>% 
  count(arr_delay) %>% 
  mutate(prop = n/sum(n))

# skim() 
flight_data %>% 
  skim() 

flight_data %>% 
  skim(dest, carrier) 

# train / test data 나누기 
set.seed(2022)

# Put 3/4 of the data into the training set , 1/4 if the data into the testing set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

# 음식을 만들때 요리 방법 설명서 
# 머신러닝 전처리 방법 설명서 
# 종속 변수 (y) -- arr_delay 설정 / 독립변수(x) -- arr_delay 빼고 다 사용!
# 총 10개의 입력 변수 중 1개의 종속변수 , 9개의 독립변수가 존재 
flights_rec <-  recipe(arr_delay ~ ., data = train_data) %>% 
                update_role(flight, time_hour, new_role = "ID") %>%  # flight, time_hour 두 개의 변수가 겹치는 것이 있으므로 unique 한 변수를 만들기 위해 앞과 같이 전처리함.  
                step_date(date, features = c("dow", "month")) %>%               
                step_holiday(date, 
                             holidays = timeDate::listHolidays("US"), 
                             keep_original_cols = FALSE) %>% 
                step_mutate(my_flight = flight + 3) %>%  # tidymodels 에서는 전처리할 때 앞에 step_라는 변수명이 있어야한다. 
                step_dummy(all_nominal_predictors()) %>%
                step_zv(all_predictors()) %>% 
                prep() # 레시피의 끝 

# 위의 레시피 방법을 기준으로 train_data , test_data을 bake()함수로 전처리 
train_data2 <- bake(flights_rec, train_data) 
test_data2 <- bake(flights_rec, test_data)

train_data2 %>% glimpse()
test_data2 %>% glimpse()

# 모델 만들기 
# 회귀모델 :linear_reg() %>% set_engine('lm')

# Logistic Regression Model
lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")

# 워크플로우 
flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)

# 모델 학습 
flights_fit <- 
  flights_wflow %>% 
  fit(data = train_data)

# 모델 결과를 tibble로 깔끔하게 보여준다. 
flights_fit %>% 
  extract_fit_parsnip() %>% 
  tidy()

test_data %>% glimpse() 

predict(flights_fit, test_data, type = "prob")
flights_aug <- 
  augment(flights_fit, test_data)

# The data look like: 
flights_aug %>%
  select(arr_delay,
         time_hour,
         flight,
         .pred_class,
         .pred_on_time)

flights_aug %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()
