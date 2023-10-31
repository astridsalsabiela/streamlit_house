import pickle
import streamlit as st

model = pickle.load(open('price-prediction.sav','rb'))

st.title('Estimasi Harga Rumah')

bedrooms: st.number_input('Input Jumlah kamar')
bathrooms: st.number_input('Input Jumlah kamar mandi di dalam rumah')
sqft_living: st.number_input('Input Luas area rumah')
sqft_lot: st.number_input ('Input Luas area tanah')
floors: st.number_input ('Input Jumlah lantai rumah')
condition: st.number_input ('Input kondisi rumah')
grade: st.number_input ('Input Peringkat kualitas rumah')
yr_built: st.number_input ('Input Tahun pembangunan rumah')

predict = ''

if st.button('Estimasi Harga'):
    predict =model.predict(
        [[bedrooms,bathrooms,sqft_living,sqft_lot,floors,condition,grade,yr_built]]
    )
    st.write ('Estimasi harga rumah dalam Rupe :', predict)