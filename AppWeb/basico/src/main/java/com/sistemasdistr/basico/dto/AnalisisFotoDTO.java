package com.sistemasdistr.basico.dto;

import lombok.Data;

@Data
public class AnalisisFotoDTO {
    private int conteo;           // Número de varroas detectadas
    private String rutaMarcada;   // Ruta de la imagen con los recuadros rojos
    private String nombreArchivo; // Nombre original para identificarla
}