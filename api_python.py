import json
import os
import time

from flask import Flask, jsonify, request

app = Flask(__name__)

dados = [
    {"id": 1, "nome": "Produto 1", "preco": 50},
    {"id": 2, "nome": "Produto 2", "preco": 150}
]

# Rota para retornar todos os produtos (GET)
@app.route('/produtos', methods=['GET'])
def get_produtos(): 
    return jsonify(dados)

# Rota para buscar um produto específico por ID (GET)
@app.route('/produtos/<int:id>', methods=['GET'])
def get_produto(id):
    produto = next((item for item in dados if item['id'] == id), None)
    if produto: return jsonify(produto)
    else: return jsonify({"erro": "Produto não encontrado"}), 404

# Rota para criar um novo produto (POST)
@app.route('/produtos', methods=['POST'])
def add_produto():
    novo_produto = request.json
    dados.append(novo_produto)
    return jsonify(novo_produto), 201

# Rota para atualizar um produto existente (PUT)
@app.route('/produtos/<int:id>', methods=['PUT'])
def update_produto(id):
    produto = next((item for item in dados if item['id'] == id), None)
    if produto:
        produto.update(request.json)
        return jsonify(produto)
    else:
        return jsonify({"erro": "Produto não encontrado"}), 404

# Rota para deletar um produto (DELETE)
@app.route('/produtos/<int:id>', methods=['DELETE'])
def delete_produto(id):
    global dados
    dados = [item for item in dados if item['id'] != id]
    return jsonify({"message": "Produto deletado"}), 200

if __name__ == '__main__':
    app.run(debug=False)
