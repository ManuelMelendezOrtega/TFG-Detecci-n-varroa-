package com.tfg.basico.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDate;
import java.util.List;

/**
 * Entidad que representa la tabla "sesion_muestreo" en la base de datos.
 * Funciona como un cuaderno de campo virtual que agrupa todas las fotos 
 * analizadas en una misma revisión de las colmenas.
 */
@Entity
@Data 
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "sesion_muestreo")
public class SesionMuestreo {

    // Identificador único (Clave Primaria) de la sesión de control
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    // Relación Many-To-One: Muchos muestreos diferentes pueden pertenecer al mismo apicultor.
    // Se usa FetchType.LAZY para no sobrecargar la memoria del servidor de forma innecesaria.
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "usuario_id", nullable = false)
    private User usuario;

    // Fecha exacta en la que el apicultor realizó el control en el apiario
    @Column(nullable = false)
    private LocalDate fecha;

    // Cantidad total de imágenes que se subieron y procesaron en esta sesión
    @Column(name = "num_fotos", nullable = false)
    private Integer numFotos;

    // El promedio de ácaros Varroa encontrados por foto (clave para saber el nivel de gravedad)
    @Column(name = "media_varroas", nullable = false)
    private Float mediaVarroas;

    // Relación One-ToMany: Una sesión contiene una lista con muchas fotos detalladas.
    // 'cascade = CascadeType.ALL' significa que si borramos esta sesión, 
    // automáticamente se borrarán de la base de datos todas las fotos que tenía dentro.
    @OneToMany(mappedBy = "sesion", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<FotoDetalle> fotos;
}