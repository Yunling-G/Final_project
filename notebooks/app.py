import streamlit as st
from PIL import Image
from sentiment_generator_v2 import sentiment_generator

def main():
	image = Image.open('logo.png')
	st.image(image)
	st.title('Sentiment Generator')
	st.subheader('Generate sentiment according to your comment about ChatGPT')
	menu = ['Customer Experience Center']
	choice = st.sidebar.selectbox('Menu',menu)
	text = st.text_input('Please enter your comment')

	if st.button("Start"):
		output = sentiment_generator(text)
		st.text(output)
if __name__ == '__main__':
	main()
