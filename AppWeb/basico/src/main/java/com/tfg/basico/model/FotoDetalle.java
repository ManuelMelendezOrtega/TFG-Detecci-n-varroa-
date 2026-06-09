package com.tfg.basico.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Entidad que representa la tabla "foto_detalle" en la base de datos.
 * Guarda la información individualizada de cada imagen analizada por la IA.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "foto_detalle")
public class FotoDetalle {

    // Identificador único (Clave Primaria) autoincremental en la base de datos
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    // Relación Many-To-One: Muchas fotos pueden pertenecer a una misma sesión de muestreo.
    // Se usa FetchType.LAZY para cargar los datos de la sesión solo cuando sea estrictamente necesario.
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "sesion_id", nullable = false)
    private SesionMuestreo sesion;

    // Número de ácaros Varroa detectados específicamente en esta fotografía
    @Column(name = "conteo_varroas", nullable = false)
    private Integer conteoVarroas;

    // Nombre único del archivo de imagen que subió originalmente el apicultor
    @Column(name = "ruta_imagen_original")
    private String rutaImagenOriginal;

    // Nombre del archivo generado por Python que contiene los recuadros verdes de la IA
    @Column(name = "ruta_imagen_marcada")
    private String rutaImagenMarcada;
}