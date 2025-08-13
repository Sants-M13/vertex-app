"""
Aplicativo Flask para ETL de dados de varejo
Prepara dados transacionais para treinamento de modelos de séries temporais no Vertex AI
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Limite de 100MB para upload

@app.route('/')
def index():
    """Serve a página HTML principal"""
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/process', methods=['POST'])
def process_data():
    """
    Rota principal para processar os arquivos de vendas e inventário (opcional)
    Executa o pipeline ETL completo e retorna o arquivo CSV processado
    """
    try:
        # 1. CARREGAMENTO E VALIDAÇÃO DOS ARQUIVOS
        
        # Verificar se o arquivo de vendas foi enviado (obrigatório)
        if 'sales_file' not in request.files:
            return jsonify({'error': 'Arquivo de vendas é obrigatório'}), 400
        
        sales_file = request.files['sales_file']
        if sales_file.filename == '':
            return jsonify({'error': 'Nenhum arquivo de vendas selecionado'}), 400
        
        # Carregar o arquivo de vendas em um DataFrame
        try:
            vendas_df = pd.read_csv(sales_file)
        except Exception as e:
            return jsonify({'error': f'Erro ao ler arquivo de vendas: {str(e)}'}), 400
        
        # Validar colunas obrigatórias do arquivo de vendas
        colunas_obrigatorias_vendas = ['timestamp', 'item_id', 'product_brand', 
                                       'product_style', 'target_quantity', 'price']
        colunas_faltantes = [col for col in colunas_obrigatorias_vendas if col not in vendas_df.columns]
        
        if colunas_faltantes:
            return jsonify({
                'error': f'Colunas obrigatórias faltando no arquivo de vendas: {", ".join(colunas_faltantes)}'
            }), 400
        
        # Verificar e carregar arquivo de inventário (opcional)
        inventario_df = None
        if 'inventory_file' in request.files:
            inventory_file = request.files['inventory_file']
            if inventory_file.filename != '':
                try:
                    inventario_df = pd.read_csv(inventory_file)
                    
                    # Validar colunas do arquivo de inventário
                    colunas_obrigatorias_inventario = ['snapshot_date', 'item_id', 'quantity_on_hand']
                    colunas_faltantes_inv = [col for col in colunas_obrigatorias_inventario 
                                            if col not in inventario_df.columns]
                    
                    if colunas_faltantes_inv:
                        return jsonify({
                            'error': f'Colunas obrigatórias faltando no arquivo de inventário: {", ".join(colunas_faltantes_inv)}'
                        }), 400
                        
                except Exception as e:
                    return jsonify({'error': f'Erro ao ler arquivo de inventário: {str(e)}'}), 400
        
        # 2. ENGENHARIA DE FEATURES PRELIMINAR
        
        # Converter timestamp para formato de data
        vendas_df['timestamp'] = pd.to_datetime(vendas_df['timestamp'])
        
        # Criar a coluna series_id concatenando brand e style
        vendas_df['series_id'] = vendas_df['product_brand'] + '_' + vendas_df['product_style']
        
        # 3. PROCESSAMENTO AGREGADO DE VENDAS
        
        # Agrupar vendas por data e series_id
        vendas_agregadas_df = vendas_df.groupby(['timestamp', 'series_id']).agg({
            'target_quantity': 'sum',  # total_quantity_sold
            'price': lambda x: (vendas_df.loc[x.index, 'price'] * 
                               vendas_df.loc[x.index, 'target_quantity']).sum() / 
                              vendas_df.loc[x.index, 'target_quantity'].sum()  # preço médio ponderado
        }).reset_index()
        
        # Renomear colunas para nomes finais
        vendas_agregadas_df.rename(columns={
            'target_quantity': 'total_quantity_sold',
            'price': 'preco_medio_praticado'
        }, inplace=True)
        
        # 4. PROCESSAMENTO CONDICIONAL DO INVENTÁRIO
        
        inventario_agregado_df = None
        
        if inventario_df is not None:
            # Converter snapshot_date para formato de data
            inventario_df['snapshot_date'] = pd.to_datetime(inventario_df['snapshot_date'])
            
            # Criar mapeamento de item_id para series_id
            mapa_id = vendas_df[['item_id', 'series_id']].drop_duplicates()
            
            # Adicionar series_id ao inventário usando o mapeamento
            inventario_df = pd.merge(inventario_df, mapa_id, on='item_id', how='left')
            
            # Remover linhas onde não conseguimos mapear o series_id
            inventario_df = inventario_df.dropna(subset=['series_id'])
            
            # Calcular total de SKUs únicos por série (para cálculo de disponibilidade)
            total_skus_por_serie_df = vendas_df.groupby('series_id')['item_id'].nunique().reset_index()
            total_skus_por_serie_df.rename(columns={'item_id': 'total_skus_contagem'}, inplace=True)
            
            # Agregar inventário por data e series_id
            inventario_agregado_df = inventario_df.groupby(['snapshot_date', 'series_id']).agg({
                'quantity_on_hand': 'sum',  # estoque_total_inicio_dia
                'item_id': lambda x: inventario_df.loc[x.index][
                    inventario_df.loc[x.index, 'quantity_on_hand'] > 0
                ]['item_id'].nunique()  # numero_modelos_disponiveis
            }).reset_index()
            
            # Renomear colunas
            inventario_agregado_df.rename(columns={
                'snapshot_date': 'timestamp',
                'quantity_on_hand': 'estoque_total_inicio_dia',
                'item_id': 'numero_modelos_disponiveis'
            }, inplace=True)
            
            # Adicionar total de SKUs para cálculo de disponibilidade
            inventario_agregado_df = pd.merge(
                inventario_agregado_df, 
                total_skus_por_serie_df, 
                on='series_id', 
                how='left'
            )
            
            # Calcular disponibilidade percentual de SKUs
            inventario_agregado_df['disponibilidade_sku_percentual'] = (
                inventario_agregado_df['numero_modelos_disponiveis'] / 
                inventario_agregado_df['total_skus_contagem']
            )
            
            # Remover coluna auxiliar
            inventario_agregado_df.drop('total_skus_contagem', axis=1, inplace=True)
        
        # 5. CRIAÇÃO DA GRADE MESTRA E CONSOLIDAÇÃO FINAL
        
        # Determinar período completo (data início e fim)
        data_inicio = vendas_df['timestamp'].min()
        data_fim = vendas_df['timestamp'].max()
        
        # Obter lista de todos os series_ids únicos
        series_ids_unicos = vendas_df['series_id'].unique()
        
        # Criar grade mestra com todas as combinações de data e series_id
        # Gerar range de datas completo
        datas_completas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
        
        # Criar produto cartesiano de datas e series_ids
        from itertools import product
        master_grid = list(product(datas_completas, series_ids_unicos))
        master_df = pd.DataFrame(master_grid, columns=['timestamp', 'series_id'])
        
        # Fazer merge com dados de vendas agregados (left join)
        master_df = pd.merge(
            master_df, 
            vendas_agregadas_df, 
            on=['timestamp', 'series_id'], 
            how='left'
        )
        
        # Se houver dados de inventário, fazer merge adicional
        if inventario_agregado_df is not None:
            master_df = pd.merge(
                master_df,
                inventario_agregado_df,
                on=['timestamp', 'series_id'],
                how='left'
            )
        
        # 6. LIMPEZA E FORMATAÇÃO DA SAÍDA
        
        # Preencher valores NaN com 0 em todas as colunas numéricas
        colunas_numericas = ['total_quantity_sold', 'preco_medio_praticado']
        
        if inventario_agregado_df is not None:
            colunas_numericas.extend(['estoque_total_inicio_dia', 
                                     'disponibilidade_sku_percentual', 
                                     'numero_modelos_disponiveis'])
        
        master_df[colunas_numericas] = master_df[colunas_numericas].fillna(0)
        
        # Definir e reordenar colunas finais baseado na presença de inventário
        if inventario_agregado_df is not None:
            colunas_finais = ['timestamp', 'series_id', 'total_quantity_sold', 
                             'preco_medio_praticado', 'estoque_total_inicio_dia', 
                             'disponibilidade_sku_percentual', 'numero_modelos_disponiveis']
        else:
            colunas_finais = ['timestamp', 'series_id', 'total_quantity_sold', 
                             'preco_medio_praticado']
        
        # Garantir que temos apenas as colunas finais na ordem correta
        master_df = master_df[colunas_finais]
        
        # Ordenar por series_id e depois por timestamp
        master_df = master_df.sort_values(['series_id', 'timestamp'])
        
        # Formatar timestamp para string no formato YYYY-MM-DD
        master_df['timestamp'] = master_df['timestamp'].dt.strftime('%Y-%m-%d')
        
        # 7. GERAÇÃO DO ARQUIVO DE SAÍDA
        
        # Converter DataFrame para CSV na memória
        output = io.StringIO()
        master_df.to_csv(output, index=False)
        output.seek(0)
        
        # Converter para bytes para envio
        output_bytes = io.BytesIO(output.getvalue().encode())
        output_bytes.seek(0)
        
        # Retornar arquivo CSV para download
        return send_file(
            output_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name='vertex_training_data.csv'
        )
        
    except Exception as e:
        # Capturar qualquer erro não tratado
        return jsonify({'error': f'Erro durante o processamento: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde do serviço"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Executar servidor Flask
    app.run(debug=True, host='0.0.0.0', port=5000)