import streamlit as st
from PIL import Image
import pandas as pd
from tensorflow.keras.datasets import mnist,fashion_mnist
from tensorflow.keras.layers import Dense,Input,Flatten,Reshape
from tensorflow.keras.models import Model




st.title("Feature Extraction using Autoencoder (AI)")

(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255


options=['Select Dataset','Mnist Dataset','FashionMnist Dataset']
selected_option=st.selectbox("Pick a Dataset want to extract features:",options)
if(selected_option=="Mnist Dataset"):
    st.write(f"selected datasets: {selected_option}")
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    #st.text_input("enter number of features extract")
    #st.dataframe(X_train[0])
elif(selected_option=='FashionMnist Dataset'):
    (X_train_f,y_train_f),(X_test_f,y_test_f)=fashion_mnist.load_data()
    X_train_f=X_train_f.astype('float32')/255
    X_test_f=X_test_f.astype('float32')/255
    st.write(f"selected dataset: {selected_option}")
    st.write(f"X_train shape: {X_train_f.shape}")
    st.write(f"y_train shape: {y_train_f.shape}")
    #st.dataframe(X_train_f[0])
    



#encodder part
input_layer_str=st.text_input("enter the size of image")
if(input_layer_str):
    try:
      input_layer_shape=tuple(map(int,input_layer_str.split(',')))
    except ValueError:
        st.error("Invalid input! Please enter dimensions as comma-seperated integer")
        input_layer_shape=None
else:
    input_layer_shape=None

    


if(input_layer_shape):
    #encoder part
    input_layer=Input(shape=input_layer_shape)
    flatten_layer=Flatten()(input_layer)
    h1=Dense(units=1024,activation='relu',kernel_initializer='he_uniform')(flatten_layer)
    h2=Dense(units=512,activation='relu',kernel_initializer='he_uniform')(h1)
    h3=Dense(units=256,activation='relu',kernel_initializer='he_uniform')(h2)
    x=st.text_input("how much feature want to get")
    if(x.isdigit()):
        x=int(x)
        bottleneck_layer=Dense(units=x,activation='relu')(h3)
        st.success(f"model created with input shape {input_layer_shape}")
        #decoder part
        h4=Dense(units=256,activation='relu',kernel_initializer='he_uniform')(bottleneck_layer)
        h5=Dense(units=512,activation='relu',kernel_initializer='he_uniform')(h4)
        h6=Dense(units=1024,activation='relu',kernel_initializer='he_uniform')(h5)
        output_layer=Dense(units=input_layer_shape[0]*input_layer_shape[1],activation='sigmoid')(h6)
        final_layer=Reshape(input_layer_shape)(output_layer)

        #autoencoder part
        autoencoder=Model(inputs=input_layer,outputs=final_layer)
        autoencoder.compile(loss='binary_crossentropy',optimizer='adam') # loss formula is based on output_layer activation is sigmoid activation is given then loss binary_crossentropy

        #train the autoencoder
        if(st.button("Train Autoencoder")):
           with st.spinner("Training the Model..."):
                history=autoencoder.fit(X_train,X_train,epochs=10,batch_size=256,validation_split=.2,verbose=1)
                st.success("Model trained successfully!")
                encoder_model=Model(inputs=input_layer,outputs=bottleneck_layer)
                X_train_new=encoder_model.predict(X_train)

                #convert extracted_featur into csv for downloading
                df_features=pd.DataFrame(X_train_new)
                csv_data=df_features.to_csv(index=False).encode("utf-8")
                st.download_button(label="Download CSV",data=csv_data,file_name="extracted_feature.csv",mime="text/csv")
                

    else:
        st.error("Please enter a valid number for the botlleneck layer")
else:
    st.warning("please enter a valid input shape for the image")






                  





