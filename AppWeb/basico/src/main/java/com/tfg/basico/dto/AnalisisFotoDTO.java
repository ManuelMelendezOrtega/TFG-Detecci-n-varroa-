package com.tfg.basico.dto;

import lombok.Data;

/**
 * Clase DTO (Data Transfer Object) para recibir la respuesta de la IA.
 * Sirve de molde para estructurar los datos del análisis que envía el microservicio de Python.
 */
@Data
public class AnalisisFotoDTO {

    // Cantidad total de ácaros Varroa detectados en la imagen
    private int conteo;           

    // Nombre del archivo físico de la imagen donde la IA ha dibujado los recuadros de detección
    private String rutaMarcada;   

    // Nombre original del archivo subido por el apicultor para su correcta identificación
    private String nombreArchivo; 
}