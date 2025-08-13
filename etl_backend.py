"""
Aplicativo Flask para ETL de dados de varejo
Prepara dados transacionais para treinamento de modelos de séries temporais no Vertex AI
"""

from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import io
from itertools import product

# Configuração da aplicação Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Limite de 100MB para upload

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """
    Serve a página HTML principal para qualquer rota, exceto as da API.
    Isso garante que o frontend carregue corretamente.
    """
    # A rota da nossa API é '/process', todas as outras devem servir o frontend
    if path != "process":
        try:
            # CORREÇÃO: Apontar para o nome de arquivo correto do frontend
            with open('etl_frontend.html', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "Frontend HTML não encontrado.", 404
    # Se a rota for '/process', o Flask continuará para a função de processamento
    # O catch_all não irá interceptá-la pois o match é feito por ordem de definição
    pass

@app.route('/process', methods=['POST'])
def process_data():
    """
    Rota principal da API para processar os arquivos de vendas e inventário (opcional).
    Executa o pipeline ETL completo e retorna o arquivo CSV processado.
    """
    try:
        # 1. CARREGAMENTO E VALIDAÇÃO DOS ARQUIVOS
        if 'sales_file' not in request.files:
            return jsonify({'error': 'Arquivo de vendas é obrigatório'}), 400
        
        sales_file = request.files['sales_file']
        if sales_file.filename == '':
            return jsonify({'error': 'Nenhum arquivo de vendas selecionado'}), 400
        
        vendas_df = pd.read_csv(sales_file)
        
        colunas_obrigatorias_vendas = ['timestamp', 'item_id', 'product_brand', 'product_style', 'target_quantity', 'price']
        if not all(col in vendas_df.columns for col in colunas_obrigatorias_vendas):
            return jsonify({'error': 'Colunas obrigatórias faltando no arquivo de vendas.'}), 400
        
        inventario_df = None
        if 'inventory_file' in request.files and request.files['inventory_file'].filename != '':
            inventory_file = request.files['inventory_file']
            inventario_df = pd.read_csv(inventory_file)
            colunas_obrigatorias_inventario = ['snapshot_date', 'item_id', 'quantity_on_hand']
            if not all(col in inventario_df.columns for col in colunas_obrigatorias_inventario):
                return jsonify({'error': 'Colunas obrigatórias faltando no arquivo de inventário.'}), 400

        # 2. ENGENHARIA DE FEATURES PRELIMINAR
        vendas_df['timestamp'] = pd.to_datetime(vendas_df['timestamp'])
        vendas_df['series_id'] = vendas_df['product_brand'].astype(str) + '_' + vendas_df['product_style'].astype(str)
        
        # 3. PROCESSAMENTO AGREGADO DE VENDAS
        # OTIMIZAÇÃO: A função lambda agora opera apenas no subgrupo 'x', o que é mais eficiente.
        def weighted_average_price(x):
            if x['target_quantity'].sum() == 0:
                return 0
            return np.average(x['price'], weights=x['target_quantity'])

        vendas_agregadas_df = vendas_df.groupby(['timestamp', 'series_id']).apply(
            lambda x: pd.Series({
                'total_quantity_sold': x['target_quantity'].sum(),
                'preco_medio_praticado': weighted_average_price(x)
            })
        ).reset_index()
        
        # 4. PROCESSAMENTO CONDICIONAL DO INVENTÁRIO
        inventario_agregado_df = None
        if inventario_df is not None:
            inventario_df['snapshot_date'] = pd.to_datetime(inventario_df['snapshot_date'])
            mapa_id = vendas_df[['item_id', 'series_id']].drop_duplicates()
            inventario_df = pd.merge(inventario_df, mapa_id, on='item_id', how='left').dropna(subset=['series_id'])
            
            total_skus_por_serie_df = vendas_df.groupby('series_id')['item_id'].nunique().reset_index().rename(columns={'item_id': 'total_skus_contagem'})
            
            inventario_agregado_df = inventario_df.groupby(['snapshot_date', 'series_id']).agg(
                estoque_total_inicio_dia=('quantity_on_hand', 'sum'),
                numero_modelos_disponiveis=('item_id', lambda x: inventario_df.loc[x.index][inventario_df.loc[x.index, 'quantity_on_hand'] > 0]['item_id'].nunique())
            ).reset_index().rename(columns={'snapshot_date': 'timestamp'})
            
            inventario_agregado_df = pd.merge(inventario_agregado_df, total_skus_por_serie_df, on='series_id', how='left')
            inventario_agregado_df['disponibilidade_sku_percentual'] = (inventario_agregado_df['numero_modelos_disponiveis'] / inventario_agregado_df['total_skus_contagem']).fillna(0)
            inventario_agregado_df.drop('total_skus_contagem', axis=1, inplace=True)
        
        # 5. CRIAÇÃO DA GRADE MESTRA E CONSOLIDAÇÃO FINAL
        data_inicio, data_fim = vendas_df['timestamp'].min(), vendas_df['timestamp'].max()
        series_ids_unicos = vendas_df['series_id'].unique()
        datas_completas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
        
        master_df = pd.DataFrame(product(datas_completas, series_ids_unicos), columns=['timestamp', 'series_id'])
        master_df = pd.merge(master_df, vendas_agregadas_df, on=['timestamp', 'series_id'], how='left')
        
        if inventario_agregado_df is not None:
            master_df = pd.merge(master_df, inventario_agregado_df, on=['timestamp', 'series_id'], how='left')
        
        # 6. LIMPEZA E FORMATAÇÃO DA SAÍDA
        colunas_finais = ['timestamp', 'series_id', 'total_quantity_sold', 'preco_medio_praticado']
        if inventario_agregado_df is not None:
            colunas_finais.extend(['estoque_total_inicio_dia', 'disponibilidade_sku_percentual', 'numero_modelos_disponiveis'])
        
        for col in colunas_finais:
            if col not in master_df.columns:
                master_df[col] = 0
        
        master_df[colunas_finais] = master_df[colunas_finais].fillna(0)
        master_df = master_df[colunas_finais].sort_values(['series_id', 'timestamp'])
        master_df['timestamp'] = master_df['timestamp'].dt.strftime('%Y-%m-%d')
        
        # 7. GERAÇÃO DO ARQUIVO DE SAÍDA
        output = io.BytesIO()
        master_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='vertex_training_data.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Erro interno do servidor durante o processamento: {str(e)}'}), 500

# REMOÇÃO: O bloco if __name__ == '__main__' foi removido pois a Vercel
# cuida de iniciar o servidor. Este bloco é apenas para testes locais.
