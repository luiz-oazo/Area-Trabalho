#!/usr/bin/env python
# coding: utf-8

from flask import Flask, jsonify, request
import oracledb
import json

app = Flask(__name__)

# Função para conectar ao banco de dados Oracle
def get_db_connection():
    DATA_FILE = "oracle_conn.json"
    file = open(DATA_FILE, 'r')
    xfile = json.load(file)

    connection = oracledb.connect(
        user=xfile[0]["user"],
        password=xfile[0]["password"],
        dsn=xfile[0]["dsn"] 
    )
    return connection

# Rota para retornar todos os funcionarios (GET)
@app.route('/func', methods=['GET'])
def get_funcs():
    connection = get_db_connection()
    cursor = connection.cursor()

    id = request.args.get('id')
    nome = request.args.get('nome')
    idade = request.args.get('idade')
    empresa = request.args.get('empresa')
    salario = request.args.get('salario')
    
    # print(id, nome, idade, empresa, salario)
    
    query = "SELECT * FROM func WHERE 1=1"
    params = []
    
    if id:
        query += " AND ID = :id"
        params.append(id)
    if nome:
        query += " AND nome = :nome"
        params.append(nome)
    if idade:
        query += " AND idade = :idade"
        params.append(idade)
    if empresa:
        query += " AND empresa = :empresa"
        params.append(empresa)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    if rows:
        funcionario = [{"id": row[0], "nome": row[1], "idade":  row[2], "empresa":  row[3], "salario":  row[4]} for row in rows] 
        return jsonify(funcionario)
    else:
        return jsonify({"message": "Funcionario não encontrado"}), 404
    
# Rota para buscar um produto específico por ID (GET)
@app.route('/func/<int:id>', methods=['GET'])
def get_func(id):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Consulta SQL para buscar um produto por ID
    cursor.execute("SELECT * FROM func WHERE id = :id", [id])
    row = cursor.fetchone()
    
    cursor.close()
    connection.close()
    
    if row:
        funcionario = [{"id": row[0], "nome": row[1], "idade":  row[2], "empresa":  row[3], "salario":  row[4]}] 
        return jsonify(funcionario)
    else:
        return jsonify({"message": "Funcionario não encontrado"}), 404

# Rota para adicionar um novo produto (POST)
@app.route('/func', methods=['POST'])
def add_func():
    novo_func = request.json
    nome = novo_func.get('nome')
    idade = novo_func.get('idade')
    empresa = novo_func.get('empresa')
    salario = novo_func.get('salario')
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Consulta SQL para inserir um novo produto
    cursor.execute("""
        INSERT INTO func (nome, idade, empresa, salario)
        VALUES (:nome, :idade, :empresa, :salario)
    """, [nome, idade, empresa, salario])
    
    connection.commit()  # Confirma a transação
    cursor.close()
    connection.close()
    
    return jsonify({"message": "Funcionario adicionado"}), 201

# Rota para atualizar um produto existente (PUT)
@app.route('/func/<int:id>', methods=['PUT'])
def update_func(id):
    func_atualizado = request.json
    nome = func_atualizado.get('nome')
    salario = func_atualizado.get('salario')
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Consulta SQL para atualizar um produto
    cursor.execute("""
        UPDATE func
        SET nome = :nome, salario = :salario
        WHERE id = :id
    """, [nome, salario, id])
    
    connection.commit()  # Confirma a transação
    cursor.close()
    connection.close()
    
    return jsonify({"message": "Funcionario atualizado"}), 200

# Rota para deletar um produto (DELETE)
@app.route('/func/<int:id>', methods=['DELETE'])
def delete_func(id):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Consulta SQL para deletar um produto por ID
    cursor.execute("DELETE FROM func WHERE id = :id", [id])
    
    connection.commit()  # Confirma a transação
    cursor.close()
    connection.close()
    
    return jsonify({"message": "Funcionario removido"}), 200

if __name__ == '__main__':
    app.run(debug=False)






