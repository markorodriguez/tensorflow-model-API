from src.service.train_model import use_model

def predict_text(params): 
    response = []
    model = use_model()

    for param in params:
        global qualification

        new_data = [param['textMessage']]
        
        prediction = model.predict(new_data)

        if prediction[0] > 0.5:
            #print('Positivo')
            param["qualification"] = 'Positivo'
        else:
            param["qualification"] = 'Negativo'

        response.append(param)

    return response
