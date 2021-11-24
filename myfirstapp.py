import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Welcome To My First Application!")
st.write("Creator: Amira")

option = st.sidebar.selectbox(
    'Select a models',
     ['Choose:', 'Linear Regression', 'Logistic Regression','Analysis Tool'])

if  option=='Choose:':
    from PIL import Image
    image0 = Image.open('da.jpg')
    st.image(image0)
    st.write("(image from www.sisense.com)")

elif  option=='Linear Regression':
    
    st.write("Simple Linear Regression.")
    st.write("This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires.")
    st.write("Ref: https://archive.ics.uci.edu/ml/datasets/student+performance")
    
    import math
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import pandas as pd
    
    data = pd.read_csv('student_mat.csv', sep=';')
    st.dataframe(data)

    # we look at g3 becuse it is the final grade
    X1 = data.studytime
    y1 = data.G3

    # arrange into a nice array
    X1 = X1[:, np.newaxis]

    # call linear reg
    model = LinearRegression(fit_intercept=True)
    model.fit(X1, y1) 
    y1_pred = model.predict(X1)

    from PIL import Image
    image = Image.open('fig.png')
    st.image(image)
    
    st.write("Based on the chart above, it does not look like a good line.")
    # using rmse. the lesser, the better
    st.write(" RMSE: {}.".format(math.sqrt(mean_squared_error(y1, y1_pred))))
    st.write(" The model coefficient : {}.".format(model.coef_))
    st.write(" The model intercept : {}.".format(model.intercept_))

    X2 = data.traveltime
    y2 = data.G3
    X2 = X2[:, np.newaxis]
    model.fit(X2, y2) 
    y2_pred = model.predict(X2)
    
    from PIL import Image
    image1 = Image.open('fig1.png')
    st.image(image1)
    
    st.write("Based on the chart above, it is not a good result too since it is almost similar like the one before")
    st.write(" RMSE: {}.".format(math.sqrt(mean_squared_error(y2, y2_pred))))

    X3 = data[["studytime","traveltime"]]
    y3 = data.G3
    model.fit(X3, y3) 
    y3_pred = model.predict(X3)
    
    from PIL import Image
    image2 = Image.open('fig2.png')
    st.image(image2)
    
    st.write("Figure above is to help us to see the pattern of the travel time based on study time.")
    st.write(" RMSE: {}.".format(math.sqrt(mean_squared_error(y3, y3_pred))))
    
    data1 = data.groupby(["studytime","traveltime"]).mean()
    st.write("Table below shows the mean based on study time and travel time.")
    data1 = data.groupby(["studytime","traveltime"]).mean()
    st.dataframe(data1)

    data1 = data.groupby(["studytime","traveltime"]).mean().reset_index()
    plt.scatter(data1.studytime, data1.traveltime, s=2*(data1.G3)**3 , alpha=0.5)
    
    from PIL import Image
    image3 = Image.open('fig3.png')
    st.image(image3)
    
elif option=='Logistic Regression':
    st.write("Logistic Regression")
    st.write("This dataset is from the National Institute of Diabetes and Digestive and Kidney Diseases.")
    st.write("Ref: https://www.kaggle.com/uciml/pima-indians-diabetes-database")
    
    #import pandas
    import pandas as pd
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    pima = pd.read_csv("diabetes.csv", header = 0, names = col_names)
    st.dataframe(pima)

    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
    X = pima[feature_cols] 
    y = pima.label 
    
    # split X and y into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # import the metrics class
    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # display
    option = st.selectbox(
        'Model evaluation using Confusion Matrix:',
        ('Accuracy', 'Precision', 'Recall', 'ROC AUC'))
    st.write('You selected:', option)

    if  option=='Accuracy':
      from sklearn.metrics import accuracy_score
      st.write("The accuracy score: ", accuracy_score(y_test, y_pred))

    elif  option=='Precision':
      from sklearn.metrics import precision_score
      st.write("The accuracy score: ", precision_score(y_test, y_pred))

    elif  option=='Recall':
      from sklearn.metrics import recall_score
      st.write("The recall score: ", recall_score(y_test, y_pred))

    elif  option=='ROC AUC':
      from sklearn.metrics import roc_auc_score
      st.write("The ROC AUC score: ", roc_auc_score(y_test, y_pred))

elif  option=='Analysis Tool':
  st.write("#### A simple interactive analysis tools")
  st.write("This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires.")
  st.write("Data Ref: https://archive.ics.uci.edu/ml/datasets/student+performance")

  import pandas as pd
  import numpy as np
  import streamlit as st
    
  st.write("Table below shows the original data.")
  df = pd.read_csv('student_mat.csv', sep = ";")
  st.dataframe(df)

  positions = list(df['school'].drop_duplicates())
  teams = list(df['age'].drop_duplicates())

  st.sidebar.write("Filter Choices:")
    
  position_choice = st.sidebar.multiselect(
      'Type of school:', positions, default=positions)
  teams_choice = st.sidebar.multiselect(
      "Age:", teams, default=teams)
  
  df = df[df['school'].isin(position_choice)]
  df = df[df['age'].isin(teams_choice)]

  st.title(f"Student Analysis")
  st.markdown("Student dataframe based on your filter choices.")
  st.dataframe(df.sort_values('age',
              ascending=False).reset_index(drop=True))

  st.markdown('### Plotting:')
  st.write("This is our plot.")
  st.vega_lite_chart(df, {
      'mark': {'type': 'circle', 'tooltip': True},
      'encoding': {
          'x': {'field': 'age', 'type': 'quantitative'},
          'y': {'field': 'absences', 'type': 'quantitative'},
          'color': {'field': 'school', 'type': 'nominal'},
          'tooltip': [{"field": 'famsize', 'type': 'nominal'}, {'field': 'age', 'type': 'quantitative'}, {'field': 'traveltime', 'type': 'quantitative'}],
      },
      'width': 700,
      'height': 400,
  })

  st.vega_lite_chart(df, {
      'mark': {'type': 'circle', 'tooltip': True},
      'encoding': {
          'x': {'field': 'age', 'type': 'quantitative'},
          'y': {'field': 'health', 'type': 'quantitative'},
          'color': {'field': 'school', 'type': 'nominal'},
          'tooltip': [{"field": 'reason', 'type': 'nominal'}, {'field': 'freetime', 'type': 'quantitative'}, {'field': 'studytime', 'type': 'quantitative'}],
      },
      'width': 700,
      'height': 400,
  })
  st.sidebar.write("Ref: https://fcpython.com/data-analysis/building-interactive-analysis-tools-with-python-streamlit")
