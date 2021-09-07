import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.cluster import KMeans
from multiprocessing import Process
import matplotlib.pyplot as plt
import seaborn as sns
import SessionState
import time
import psutil

st.sidebar.title("Controls")
stop = st.sidebar.button("Stop")

state = SessionState.get(pid=None)


def job():
    for _ in range(100):
        print("In progress")
        time.sleep(1)

if stop:
    p = psutil.Process(state.pid)
    p.terminate()
    st.write("Stopped process with pid:", state.pid)
    state.pid = None

#####################

st.write("""
# Previsão depósitos a prazo
""")

st.markdown("""
<style>
body {
    color: #466e;
    etc. 
}
</style>
    """, unsafe_allow_html=True)

st.sidebar.header('Parâmetros')
# sidebar(inicio,fim,defaut_value)

def user_input_features():

    df1 = pd.read_csv(r"C:\Users\Zé\Downloads\Downloads\Um\Sino\bankFull_convertion_without_unknown.csv", sep = ",")

    age = st.sidebar.slider('Age', 1, 10, 99)
##################################
    valores_job = ("admin.","blue-collar","entrepreneur","housemaid","management"
        ,"retired","self-employed","services","student","technician","unemployed","unknown")
    job_lista = list(valores_job)

    job = st.sidebar.selectbox("Job", valores_job)
##################################
    valores_marital = ("divorced","married","single")
    marital_lista = list(valores_marital)

    marital = st.sidebar.selectbox("Marital", valores_marital)
#######################################
    valores_education = ("primary","secondary","tertiary","unknown")
    education_lista = list(valores_education)

    education = st.sidebar.selectbox("Education", valores_education)
#######################################
    valores_default = ("No","Yes")
    default_lista = list(range(len(valores_default)))

    default = st.sidebar.selectbox("Default", default_lista, format_func=lambda x: valores_default[x])
#######################################
    balance = st.sidebar.number_input('Balance')
#######################################
    valores_housing = ("No","Yes")
    job_housing = list(range(len(valores_housing)))

    housing = st.sidebar.selectbox("Housing", job_housing, format_func=lambda x: valores_housing[x])
#######################################
    valores_loan = ("No","Yes")
    loan_lista = list(range(len(valores_loan)))

    loan = st.sidebar.selectbox("Loan", loan_lista, format_func=lambda x: valores_loan[x])
#######################################
    valores_contact = ("cellular","telephone","unknown")
    contact_lista = list(valores_contact)

    contact = st.sidebar.selectbox("Contact", valores_contact)
#######################################
    day = st.sidebar.slider('Day', 1, 1, 31)
#######################################
    valores_month = ("April","August","December","February",
        "January","July","June","March","May","November","October","September")
    month_lista = list(range(len(valores_month)))

    month = st.sidebar.selectbox("Month", month_lista, format_func=lambda x: valores_month[x])
#######################################
    campaign = st.sidebar.slider('Campaign', 0, 0, 2)
 #######################################
    pdays = st.sidebar.slider('Pdays', -1, 0, 100)
#######################################
    previous = st.sidebar.slider('Previous', -1, 0, 10)
#######################################
    valores_poutcome = ("failure","other","success","unknown")
    poutcome_lista = list(valores_poutcome)

    poutcome = st.sidebar.selectbox("Poutcome", valores_poutcome)
#######################################

    data = {'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,}
 
    df = pd.DataFrame(data, index=[0])
    df['marital'] = df['marital'].astype('category')
    df['education'] = df['education'].astype('category')
    df['job'] = df['job'].astype('category')
    df['contact'] = df['contact'].astype('category')
    df['poutcome'] = df['poutcome'].astype('category')
    df = pd.get_dummies(df, columns=['marital'], prefix = ['marital'])
    df = pd.get_dummies(df, columns=['education'], prefix = ['education'])
    df = pd.get_dummies(df, columns=['job'], prefix = ['job'])
    df = pd.get_dummies(df, columns=['contact'], prefix = ['contact'])
    df = pd.get_dummies(df, columns=['poutcome'], prefix = ['poutcome'])

    kmeans_data= pd.DataFrame(df, index=[0],columns=df1.columns)
    kmeans_data.fillna(0, inplace=True)

    features = pd.DataFrame(df, index=[0],columns=df1.columns)
    
    features.drop(['y','duration','balance'] , axis=1 , inplace=True)
    features.fillna(0, inplace=True)

    z = []
    z.append(features)
    z.append(data)
    z.append(kmeans_data)
    return z

df1 = pd.read_csv(r"C:\Users\Zé\Downloads\Downloads\Um\Sino\bankFull_convertion_without_unknown.csv", sep = ",")

t = user_input_features()
df_kmeans = t[2]
df = t[0]
df_data = t[1]
df_show = pd.DataFrame([df_data])
df_show_2 = pd.concat([df_show["age"],df_show["job"],df_show["marital"],df_show["education"],df_show["default"],df_show["balance"],df_show["housing"],df_show["loan"]], axis=1)
df_show_3 = pd.concat([df_show["contact"],df_show["day"],df_show["month"]], axis=1)
df_show_4 = pd.concat([df_show["campaign"],df_show["pdays"],df_show["previous"],df_show["poutcome"]], axis=1)
#df_show_4 = pd.concat([df["education_primary"],df["education_secondary"],df["education_tertiary"]], axis=1)

#df_show_5 = pd.concat([df["job_admin."],df["job_blue-collar"],df["job_entrepreneur"],df["job_housemaid"],df["job_management"]], axis=1)
#df_show_6 = pd.concat([df["job_self-employed"],df["job_services"],df["job_student"],df["job_technician"],df["job_unemployed"]], axis=1)

#df_show_7 = pd.concat([df["contact_cellular"],df["contact_telephone"],df["housing"]], axis=1)
#df_show_8 = pd.concat([df["poutcome_failure"],df["poutcome_other"],df["poutcome_success"]], axis=1)


#,"default","housing","loan","day","month","campaign","pdays","previous",  8
#"marital_divorced"   , "marital_married", "marital_single",  "education_primary" ,  "education_secondary" ,5
#"education_tertiary",  "job_admin." , "job_blue-collar", "job_entrepreneur" ,   "job_housemaid" ,  "job_management"  ,6
#"job_retired" ,"job_self-employed" ,  "job_services"  ,  "job_student", "job_technician" , "job_unemployed",  "contact_cellular" ,  7 
#"contact_telephone" ,  "poutcome_failure" ,   "poutcome_other" , "poutcome_success"]4

st.subheader('Dados do cliente')
st.table(df_show_2)

st.subheader('Dados do último contacto (ou chamada) relativos à campanha atual')
st.table(df_show_3)

st.subheader('Dados complementares (ou outros atributos)')
st.table(df_show_4)
st.progress(100)
#if st.button("Gráfico"):
#    import altair as alt
#    chart = alt.Chart(df1).mark_bar().encode(
#    alt.X("age", bin=True),
#    y='count()',)
#    st.altair_chart(chart)



#st.dataframe(df_show_3)
#st.dataframe(df_show_4)
#st.dataframe(df_show_5)
#st.dataframe(df_show_6)
#st.dataframe(df_show_7)
#st.dataframe(df_show_8)


X = df1.drop(['y','duration','balance'] , axis=1)
y = df1["y"]
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


if st.button("Prever com MLPClassifier"):

    clf = MLPClassifier(random_state=1, max_iter=1000)
    clf.fit(X_train, y_train)

    prediction = clf.predict(df)
    prediction_proba_yes = clf.predict_proba(df)[:, 1]
    prediction_proba_no = clf.predict_proba(df)[:, 0]

    prediction_proba_yes = prediction_proba_yes*100
    prediction_proba_yes= np.around(prediction_proba_yes,2)
    prediction_proba_no = prediction_proba_no*100
    prediction_proba_no= np.around(prediction_proba_no,2)
    if(prediction == 0):
        st.info('É provável que o cliente responda Não!')

    if(prediction == 1):
        st.info('É provável que o cliente responda Sim!')

    st.info('A previsão de dizer Sim é {}%'.format(prediction_proba_yes))
    st.info('A previsão de dizer Não é {}%'.format(prediction_proba_no))


if st.button("Prever com LinearDiscriminantAnalysis"):

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    prediction = clf.predict(df)
    prediction_proba_yes = clf.predict_proba(df)[:, 1]
    prediction_proba_no = clf.predict_proba(df)[:, 0]

    prediction_proba_yes = prediction_proba_yes*100
    prediction_proba_yes= np.around(prediction_proba_yes,2)
    prediction_proba_no = prediction_proba_no*100
    prediction_proba_no= np.around(prediction_proba_no,2)
    if(prediction == 0):
        st.info('É provável que o cliente responda Não!')

    if(prediction == 1):
        st.info('É provável que o cliente responda Sim!')

    st.info('A previsão de dizer Sim é {}%'.format(prediction_proba_yes))
    st.info('A previsão de dizer Não é {}%'.format(prediction_proba_no))

if st.button("Prever com Arvore de decisão"):

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    prediction = clf.predict(df)
    prediction_proba_yes = clf.predict_proba(df)[:, 1]
    prediction_proba_no = clf.predict_proba(df)[:, 0]

    prediction_proba_yes = prediction_proba_yes*100
    prediction_proba_yes= np.around(prediction_proba_yes,2)
    prediction_proba_no = prediction_proba_no*100
    prediction_proba_no= np.around(prediction_proba_no,2)
    if(prediction == 0):
        st.info('É provável que o cliente responda Não!')

    if(prediction == 1):
        st.info('É provável que o cliente responda Sim!')

    st.info('A previsão de dizer Sim é {}%'.format(prediction_proba_yes))
    st.info('A previsão de dizer Não é {}%'.format(prediction_proba_no))


with st.expander(label="Upload Ficheiro de dados", expanded=False):
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file,sep=",")
        upload_df.drop(['y','duration','balance'] , axis=1 , inplace=True)

        clf = MLPClassifier(random_state=1, max_iter=1000)
        clf.fit(X_train, y_train)


        prediction_proba_yes = clf.predict_proba(upload_df)[:, 1]
        prediction_proba_yes = prediction_proba_yes*100

    Numero_head = st.number_input('Percentagem de linhas',min_value=0.0,max_value=100.0)

    if st.button("Prever com MLPC"):

        num_rows = len(upload_df.index)
        num_rows= int(num_rows)
        x=(num_rows*Numero_head)/100
        x = int(x)


        teste=pd.DataFrame(prediction_proba_yes)
        teste['Id'] = teste.index

        teste.columns = ['Previsão_Sim','Id Cliente']

        teste.sort_values(by='Previsão_Sim', ascending=False,inplace=True)

        teste.index = range(teste.shape[0])
        teste=teste.head(x)
        st.table(teste)


with st.expander(label="Clusters", expanded=False):

    Numero_clusters = st.number_input('Número de Clusters',min_value=2.0,max_value=10.0)
    Numero_clusters = int(Numero_clusters)
#######
    valores_x = ("balance","age","month","housing","campaign","pdays","previous")
    x_lista = list(valores_x)

    x_cluster =st.selectbox("Escolha o X", x_lista)
#########
    valores_y = ("age","month","balance","housing","campaign","pdays","previous")
    y_lista = list(valores_y)

    y_cluster = st.selectbox("Escolha o Y", y_lista)
########
    kmeans = KMeans(n_clusters=Numero_clusters, random_state=0)
    kmeans.fit(df1)
    predictings=kmeans.predict(df_kmeans)
######
    sns.scatterplot(x=x_cluster,y=y_cluster,data=df1, hue=kmeans.labels_, palette='Paired')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.rcParams['figure.figsize'] = [12, 12]
    st.pyplot()
    st.info('O cliente faz parte do cluster:{}'.format(predictings))

    def cluster_table_result(df):

        labels = pd.DataFrame(list(kmeans.labels_))
        labels_without_unknown = kmeans.labels_
        labels.index = range(labels.shape[0])
        labels.columns = ['clustering']
        labels_without_unknown = labels['clustering']
        labels['clustering'].astype(str).astype(int)
        df['clustering'] = labels['clustering']

        _number = list(range(Numero_clusters))
        bankFull_convertion_with_unknown_table = pd.DataFrame()
        bankFull_convertion_with_unknown_table['Clustering'] = _number
        bankFull_convertion_with_unknown_table['%y=no'] = _number
        bankFull_convertion_with_unknown_table['%y=yes'] = _number
        bankFull_convertion_with_unknown_table['Total clustering'] = _number
        for index, row in bankFull_convertion_with_unknown_table.iterrows(): 
            total_bankFull_convertion_with_unknown = df.query('clustering ==' + str(_number[index])).shape[0] 
            yes_bankFull_convertion_with_unknown = df.query('y == 1 and ' + 'clustering == ' + str(_number[index])).shape[0] * 100 / total_bankFull_convertion_with_unknown
            no_bankFull_convertion_with_unknown = df.query('y == 0 and ' + 'clustering == ' + str(_number[index])).shape[0] * 100 / total_bankFull_convertion_with_unknown
            bankFull_convertion_with_unknown_table.loc[index, '%y=yes'] = yes_bankFull_convertion_with_unknown
            bankFull_convertion_with_unknown_table.loc[index, '%y=no'] = no_bankFull_convertion_with_unknown
            bankFull_convertion_with_unknown_table.loc[index, 'Total clustering'] = (total_bankFull_convertion_with_unknown)
            bankFull_convertion_with_unknown_table = bankFull_convertion_with_unknown_table.rename(columns={'clustering' : 'Nº de Clusters'})
        return bankFull_convertion_with_unknown_table
    table = cluster_table_result(df1)
    st.write(table)


    def metricas(df):

        labels = pd.DataFrame(list(kmeans.labels_))
        labels_without_unknown = kmeans.labels_
        labels.index = range(labels.shape[0])
        labels.columns = ['clustering']
        labels_without_unknown = labels['clustering']
        labels['clustering'].astype(str).astype(int)
        df['clustering'] = labels['clustering']

        silhouette = metrics.silhouette_score(df,labels_without_unknown, metric='euclidean')
        davies_bouldin = metrics.davies_bouldin_score(df, labels_without_unknown)

        z = []
        z.append(silhouette)
        z.append(davies_bouldin)

        return z

    if st.button("Avaliação Silhouette"):
        t = metricas(df1)
        silhouette = t[0]
        st.write(silhouette)
    if st.button("Avaliação Davies Bouldin"):
        t = metricas(df1)
        davies_bouldin = t[1]
        st.write(davies_bouldin)



with st.expander(label="Dados Perfil Cluster", expanded=False):
    def clustering_profiles(number, dataset, yes_no):
        clustering_profiles_table = dataset.copy()
        clustering_profiles_table = clustering_profiles_table.rename(columns={'job_admin.' : 'job_admin',
        'job_blue-collar' : 'job_blue_collar', 'job_self-employed' : 'job_self_employed'})
        clustering_profiles_table = clustering_profiles_table.query('clustering == ' + str(number) + ' and y == ' + str(yes_no))
        clustering_profiles_table.index = range(clustering_profiles_table.shape[0])
        return clustering_profiles_table
        
    clustering_discription_profiles = clustering_profiles(predictings, df1, 1)

    if st.button("Age"):
       
        
        sns.histplot(x="age",data=clustering_discription_profiles, palette='Paired',bins=10)
        #st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.rcParams['figure.figsize'] = [12, 12]
        st.pyplot()
    #######################
    #job
    if st.button("Job"):
        clustering_profiles_job_admin = (clustering_discription_profiles.query('job_admin == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_blue_collar = (clustering_discription_profiles.query('job_blue_collar == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_entrepreneur = (clustering_discription_profiles.query('job_entrepreneur == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_housemaid = (clustering_discription_profiles.query('job_housemaid == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_management = (clustering_discription_profiles.query('job_management == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_retired = (clustering_discription_profiles.query('job_retired == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_self_employed = (clustering_discription_profiles.query('job_self_employed == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_services = (clustering_discription_profiles.query('job_services == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_student = (clustering_discription_profiles.query('job_student == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_technician = (clustering_discription_profiles.query('job_technician == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_job_unemployed = (clustering_discription_profiles.query('job_unemployed == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        
        cluster_column_name = ['job_admin', 'job_blue_collar',  'job_entrepreneur', 'job_housemaid',    'job_management',   'job_retired',  'job_self_employed',    'job_services', 'job_student',  'job_technician',   'job_unemployed']
        cluster_column_value = [clustering_profiles_job_admin, clustering_profiles_job_blue_collar, clustering_profiles_job_entrepreneur, clustering_profiles_job_housemaid, clustering_profiles_job_management, clustering_profiles_job_retired, clustering_profiles_job_self_employed, clustering_profiles_job_services, clustering_profiles_job_student, clustering_profiles_job_technician, clustering_profiles_job_unemployed]
        
        x_job = np.array(cluster_column_name)
        y_job = np.array(cluster_column_value)
        
        #plt.figure(figsize=(8,6))
        plt.xlabel('Quantidade')
        #plt.ylabel('coluna')
        plt.title('Jobs')
        plt.barh(x_job,y_job, color = 'b')
        st.pyplot()
    ##############################
    if st.button("Loan / Housing / Default"):
        clustering_profiles_default = (clustering_discription_profiles.query('default == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_housing = (clustering_discription_profiles.query('housing == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_loan = (clustering_discription_profiles.query('loan == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        
        clustering_profiles_default = (clustering_discription_profiles.query('default == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_housing = (clustering_discription_profiles.query('housing == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        clustering_profiles_loan = (clustering_discription_profiles.query('loan == 1').shape[0] *100) / clustering_discription_profiles.shape[0]
        cluster_column_name = ['default', 'housing', 'loan']
        cluster_column_value = [clustering_profiles_default, clustering_profiles_housing, clustering_profiles_loan]
        
        x_loan = np.array(cluster_column_name)
        y_loan = np.array(cluster_column_value)
        plt.xlabel('Quantidade') 
        #plt.ylabel('coluna') 
        plt.title('Loan / Housing / Default')
        plt.barh(x_loan, y_loan, color = 'b')
        st.pyplot()
    #defaut 
    #loan 
    #housing
    #marital    